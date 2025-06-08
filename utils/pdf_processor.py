import fitz
import os
from PIL import Image, ImageOps, ImageEnhance
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import Signal, QObject
import logging
import tensorflow as tf
from tensorflow import keras

# Tối ưu TensorFlow: bật XLA và log thiết bị
try:
    tf.config.optimizer.set_jit(True)  # Bật XLA
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device_info = f"Using GPU: {physical_devices[0].name}"
    else:
        device_info = "Using CPU (no GPU found)"
    print(device_info)
    logging.info(device_info)
except Exception as e:
    logging.warning(f"TensorFlow device config error: {e}")
    
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "pdf_classifier.keras"))
pdf_classifier = keras.models.load_model(MODEL_PATH)

class PDFProcessor(QObject):
    progress_updated = Signal(int)
    blank_pages_detected = Signal(list)
    merge_progress = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.input_folder = None  # Thêm thuộc tính input_folder để lưu đường dẫn thư mục đầu vào
        self.BLANK_PAGE_THRESHOLD = 42 * 1024  # KB
        self.WHITE_RATIO_THRESHOLD = 0.85
        self.CONTRAST_THRESHOLD = 0.1
        self.NOISE_THRESHOLD = 0.05
        self.IMAGE_SIGNIFICANCE_THRESHOLD = 0.1
        self.MIN_IMAGE_SIZE = 100 * 1024  # 100KB
        self.analysis_cache = {}
        
        logging.basicConfig(
            filename='pdf_analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('PDFProcessor')

    def calculate_image_metrics(self, image):
        """Tính toán các chỉ số của ảnh"""
        # Chuyển sang grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Chuyển đổi sang numpy array
        img_array = np.array(image)
        
        # Tính toán độ tương phản
        contrast = np.std(img_array) / 255.0
        
        # Tính toán nhiễu bằng gradient
        gradient_x = np.abs(np.diff(img_array, axis=1))
        gradient_y = np.abs(np.diff(img_array, axis=0))
        noise = (np.mean(gradient_x) + np.mean(gradient_y)) / 2 / 255.0
        
        # Tính toán histogram
        hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()  # Normalize
        
        # Tính entropy để đo độ phức tạp của ảnh
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Tính toán độ phân tán của histogram
        hist_std = np.std(hist)
        
        return {
            'contrast': contrast,
            'noise': noise,
            'entropy': entropy,
            'hist_std': hist_std
        }

    def analyze_image_significance(self, page):
        """Phân tích ý nghĩa của các ảnh trong trang"""
        image_list = page.get_images()
        if not image_list:
            return False

        total_significance = 0
        for img in image_list:
            xref = img[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Tính toán kích thước tương đối của ảnh
                page_rect = page.rect
                img_rect = page.get_image_bbox(img)
                relative_size = (img_rect.width * img_rect.height) / (page_rect.width * page_rect.height)
                
                # Tính toán các chỉ số của ảnh
                metrics = self.calculate_image_metrics(image)
                
                # Tính toán ý nghĩa của ảnh dựa trên nhiều yếu tố
                size_factor = min(1.0, len(image_bytes) / self.MIN_IMAGE_SIZE)
                contrast_factor = metrics['contrast']
                entropy_factor = min(1.0, metrics['entropy'] / 8.0)  # Normalize entropy
                hist_factor = min(1.0, metrics['hist_std'] * 10)  # Normalize histogram std
                
                # Kết hợp các yếu tố với trọng số
                significance = (
                    size_factor * 0.3 +  # Kích thước ảnh
                    contrast_factor * 0.2 +  # Độ tương phản
                    entropy_factor * 0.3 +  # Độ phức tạp
                    hist_factor * 0.2  # Độ phân tán histogram
                ) * relative_size
                
                total_significance += significance
                
                # Log chi tiết cho mỗi ảnh
                self.logger.info(f"  Image metrics:")
                self.logger.info(f"    Size: {len(image_bytes)/1024:.2f}KB")
                self.logger.info(f"    Relative size: {relative_size:.4f}")
                self.logger.info(f"    Contrast: {metrics['contrast']:.4f}")
                self.logger.info(f"    Entropy: {metrics['entropy']:.4f}")
                self.logger.info(f"    Hist std: {metrics['hist_std']:.4f}")
                self.logger.info(f"    Significance: {significance:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Lỗi khi phân tích ảnh: {str(e)}")
                continue

        return total_significance > self.IMAGE_SIGNIFICANCE_THRESHOLD

    def log_page_analysis(self, page_num, metrics):
        """Ghi log chi tiết về phân tích trang"""
        self.logger.info(f"Phân tích trang {page_num}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")

    def get_page_size(self, page):
        """Lấy kích thước của trang PDF"""
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return len(img_byte_arr.getvalue())

    def preprocess_image(self, img):
        """Tiền xử lý ảnh để cải thiện độ chính xác"""
        # Chuyển sang grayscale
        img = img.convert('L')
        # Tăng độ tương phản
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        # Làm sạch nhiễu
        img = ImageOps.autocontrast(img)
        return img

    def calculate_contrast(self, img):
        """Tính toán độ tương phản của ảnh"""
        img_array = np.array(img)
        return np.std(img_array) / 255.0

    def calculate_noise(self, img):
        """Tính toán mức độ nhiễu của ảnh"""
        img_array = np.array(img)
        # Tính gradient
        gradient_x = np.abs(np.diff(img_array, axis=1))
        gradient_y = np.abs(np.diff(img_array, axis=0))
        # Tính trung bình gradient
        noise_level = (np.mean(gradient_x) + np.mean(gradient_y)) / 2
        return noise_level / 255.0

    def is_blank_page(self, page):
        page_num = page.number + 1
        metrics = {}

        # --- Dùng model AI để phân loại trang rỗng ---
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(256/page.rect.width, 256/page.rect.height))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert('L').resize((256, 256))
            img_arr = np.array(img).astype('float32') / 255.0
            img_arr = img_arr.reshape((1, 256, 256, 1))
            pred = pdf_classifier.predict(img_arr, verbose=0)[0][0]
            # Nếu model dự đoán < 0.5 thì là rỗng, >= 0.5 là có nội dung
            if pred < 0.5:
                metrics['ai_blank'] = True
                self.log_page_analysis(page_num, metrics)
                return True
            else:
                metrics['ai_blank'] = False
        except Exception as e:
            self.logger.warning(f"AI model error: {str(e)}")

        # Kiểm tra kích thước trang
        page_size = self.get_page_size(page)
        metrics['page_size'] = f"{page_size/1024:.2f}KB"

        # Nếu kích thước trang quá nhỏ, đánh dấu là rỗng ngay lập tức
        if page_size < self.BLANK_PAGE_THRESHOLD:
            metrics['is_blank'] = True
            self.log_page_analysis(page_num, metrics)
            return True

        # Kiểm tra ý nghĩa của ảnh
        try:
            has_significant_image = self.analyze_image_significance(page)
            metrics['has_significant_image'] = has_significant_image
            if has_significant_image:
                metrics['is_blank'] = False
                self.log_page_analysis(page_num, metrics)
                return False
        except Exception as e:
            self.logger.warning(f"Lỗi khi phân tích ảnh: {str(e)}")

        # Nếu lỗi phân tích ảnh hoặc không có ảnh, dựa vào các chỉ số khác
        try:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = self.preprocess_image(img)

            # Tính toán các chỉ số
            metrics.update(self.calculate_image_metrics(img))

            # Tính tỷ lệ pixel sáng (white_ratio)
            white_pixels = sum(1 for pixel in img.getdata() if pixel > 200)
            total_pixels = pix.width * pix.height
            white_ratio = white_pixels / total_pixels
            metrics['white_ratio'] = f"{white_ratio:.4f}"

            # Logic bổ sung: Nếu white_ratio rất cao và contrast rất thấp
            if white_ratio > 0.95 and metrics['contrast'] < 0.03:
                metrics['is_blank'] = True
                self.log_page_analysis(page_num, metrics)
                return True

            # Logic cải tiến: Kết hợp thêm entropy để xác định trang rỗng
            if white_ratio > 0.995 and metrics['contrast'] < 0.03 and metrics['entropy'] < 1.5:
                metrics['is_blank'] = True
                metrics['reason'] = "High white_ratio, low contrast, and low entropy"
                self.log_page_analysis(page_num, metrics)
                return True

            # Cập nhật logging để ghi lý do không đánh dấu là rỗng
            metrics['reason'] = "Page does not meet blank criteria"
            self.log_page_analysis(page_num, metrics)

            # Kiểm tra các chỉ số để xác định trang rỗng
            if (metrics['contrast'] < 0.03 and 
                metrics['entropy'] < 0.85 and 
                metrics['noise'] < 0.06 and 
                white_ratio > 0.85):
                metrics['is_blank'] = True
                self.log_page_analysis(page_num, metrics)
                return True

        except Exception as e:
            self.logger.warning(f"Lỗi khi xử lý ảnh: {str(e)}")

        # Nếu không có đủ dữ liệu, đánh giá dựa trên các chỉ số khác
        metrics['is_blank'] = False
        self.log_page_analysis(page_num, metrics)
        return metrics['is_blank']

    def analyze_pdf(self, filepath):
        """Phân tích PDF và cache kết quả, trả về list dict gồm số trang và thumbnail"""
        if filepath in self.analysis_cache:
            return self.analysis_cache[filepath]

        doc = fitz.open(filepath)
        blank_pages = []
        total = len(doc)

        for i in range(total):
            page = doc.load_page(i)
            self.progress_updated.emit(int((i + 1) / total * 100))
            if self.is_blank_page(page):
                # Tạo thumbnail nhỏ (PNG bytes)
                pix = page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))
                img_bytes = pix.tobytes("png")
                blank_pages.append({
                    'page_num': i + 1,
                    'thumbnail': img_bytes
                })
        doc.close()
        self.analysis_cache[filepath] = blank_pages
        return blank_pages

    def remove_blank_pages(self, filepath, save_as_new=True):
        """Xóa các trang trắng dựa trên kết quả phân tích đã cache"""
        if filepath not in self.analysis_cache:
            self.analyze_pdf(filepath)

        blank_pages = self.analysis_cache[filepath]
        if not blank_pages:
            return filepath

        doc = fitz.open(filepath)

        # Nếu tệp được mã hóa, giải mã trước khi thực hiện thay đổi
        if doc.is_encrypted:
            try:
                doc.authenticate("")  # Thử giải mã với mật khẩu trống
            except Exception as e:
                raise ValueError("Không thể giải mã tệp PDF. Vui lòng kiểm tra mật khẩu.")

        # Sắp xếp danh sách blank_pages theo 'page_num'
        for page_info in sorted(blank_pages, key=lambda x: x['page_num'], reverse=True):
            doc.delete_page(page_info['page_num'] - 1)

        # Luôn lưu tệp mới
        new_path = filepath.replace(".pdf", "_cleaned.pdf")
        doc.save(new_path)
        doc.close()

        # Nếu không tạo tệp mới, thay thế tệp gốc bằng tệp mới
        if not save_as_new:
            os.replace(new_path, filepath)

        return new_path if save_as_new else filepath

    def merge_pdfs(self, folder_path):
        import os
        import re
        from pathlib import Path
        import fitz
        folder_path = Path(folder_path)
        # Nếu chính folder_path là thư mục con đúng định dạng thì xử lý luôn
        folder_name = str(folder_path).split(os.sep)[-1]
        if folder_name.startswith('A55-91-001-') and re.search(r'A55-91-001-\d{2}-\d{4}', folder_name):
            return self.merge_single_folder(str(folder_path))
        # Nếu không, tìm các thư mục con đúng định dạng
        subfolders = [f for f in folder_path.iterdir() if f.is_dir() and f.name.startswith('A55-91-001-') and re.search(r'A55-91-001-\d{2}-\d{4}', f.name)]
        if not subfolders:
            raise ValueError(f"Không tìm thấy thư mục con phù hợp trong {folder_path}")
        results = []
        for subfolder in subfolders:
            results.append(self.merge_single_folder(str(subfolder)))
        return results

    def merge_single_folder(self, folder_path, progress_callback=None):
        import os
        import fitz
        from pathlib import Path
        import re
        folder = Path(folder_path)
        folder_name = str(folder).split(os.sep)[-1]
        # Kiểm tra đúng định dạng tên thư mục
        if not (folder_name.startswith('A55-91-001-') and re.search(r'A55-91-001-\d{2}-\d{4}', folder_name)):
            raise ValueError(f"Thư mục không đúng định dạng: {folder_name}")
        pdf_files = [f for f in folder.glob('*.pdf') if f.stem.isdigit() or (f.stem != folder.name)]
        pdf_files = [f for f in pdf_files if f.name != f"{folder.name}.pdf"]
        pdf_files.sort(key=lambda x: x.stem)
        if not pdf_files:
            raise Exception("Không tìm thấy file PDF hợp lệ trong thư mục.")
        output_path = folder / f"{folder.name}.pdf"
        if output_path.exists():
            output_path.unlink()
        merged_doc = fitz.open()
        total = len(pdf_files)
        for idx, pdf_file in enumerate(pdf_files):
            doc = fitz.open(str(pdf_file))
            merged_doc.insert_pdf(doc)
            doc.close()
            if progress_callback:
                progress_callback(int((idx + 1) / total * 80))  # 0-80% cho gộp
        merged_doc.save(str(output_path))
        merged_doc.close()
        if progress_callback:
            progress_callback(90)  # 90% khi đã gộp xong
        self.analyze_pdf(str(output_path))
        if progress_callback:
            progress_callback(100)
        return str(output_path)

class QLabelLogger:
    def __init__(self, label):
        self.label = label
        self.buffer = []

    def flush(self):
        if self.buffer:
            current_text = self.label.text()
            self.label.setText(current_text + "\n" + "\n".join(self.buffer))
            self.buffer = []

    def info(self, message):
        self.buffer.append(message)
        if len(self.buffer) > 10:  # Flush khi buffer đủ lớn
            self.flush()

    def warning(self, message):
        self.info("[WARNING] " + message)

    def error(self, message):
        self.info("[ERROR] " + message)