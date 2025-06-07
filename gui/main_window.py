from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QProgressBar, QFileDialog,
                               QScrollArea, QGridLayout, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
import os
from utils.pdf_processor import PDFProcessor


class PDFWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    analysis_complete = Signal(list)

    def __init__(self, processor, filepath, save_as_new=True, analyze_only=False):
        super().__init__()
        self.processor = processor
        self.filepath = filepath
        self.save_as_new = save_as_new
        self.analyze_only = analyze_only

    def run(self):
        try:
            if self.analyze_only:
                blank_pages = self.processor.analyze_pdf(self.filepath)
                self.analysis_complete.emit(blank_pages)
            else:
                result = self.processor.remove_blank_pages(
                    self.filepath, self.save_as_new)
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MergeAndAnalyzeThread(QThread):
    progress_updated = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, processor, folder_path):
        super().__init__()
        self.processor = processor
        self.folder_path = folder_path

    def run(self):
        try:
            merged_pdf = self.processor.merge_single_folder(
                self.folder_path, progress_callback=self.progress_updated.emit)
            self.finished.emit(merged_pdf)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Tool")
        self.setMinimumSize(800, 600)
        self.processor = PDFProcessor()
        self.current_file = None
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.drop_area = QLabel("Kéo thả file PDF hoặc thư mục vào đây\nhoặc nhấn nút Chọn File/Thư mục")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 5px;
                padding: 20px;
                background: #fff;
                color: #000;
            }
        """)
        self.drop_area.setMinimumHeight(150)
        layout.addWidget(self.drop_area)

        button_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Chọn File")
        self.select_folder_btn = QPushButton("Chọn Thư mục")
        self.save_btn = QPushButton("Lưu")
        self.save_as_btn = QPushButton("Lưu thành file mới")
        self.save_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        button_layout.addWidget(self.select_file_btn)
        button_layout.addWidget(self.select_folder_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.save_as_btn)
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.thumbnails_area = QScrollArea()
        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QGridLayout(self.thumbnails_widget)
        self.thumbnails_area.setWidget(self.thumbnails_widget)
        self.thumbnails_area.setWidgetResizable(True)
        layout.addWidget(self.thumbnails_area)

    def setup_connections(self):
        self.select_file_btn.clicked.connect(self.select_file)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.save_btn.clicked.connect(lambda: self.save_pdf(False))
        self.save_as_btn.clicked.connect(lambda: self.save_pdf(True))
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        # Remove logger connection as log_area is no longer used
        # self.processor.logger = QLabelLogger(self.log_area)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if not self.select_file_btn.isEnabled():
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if not self.select_file_btn.isEnabled():
            event.ignore()
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        print(f"[DEBUG] Đường dẫn kéo thả: {path}")  # Log ra console
        import os
        import re
        # Loại bỏ dấu / ở cuối nếu có (do kéo thả thư mục trên macOS)
        if path.endswith(os.sep):
            path = path.rstrip(os.sep)
        if os.path.isfile(path) and path.lower().endswith('.pdf'):
            self.load_pdf(path)
        elif os.path.isdir(path):
            folder_name = os.path.basename(path)
            if folder_name.startswith('A55-91-001-') and re.search(r'A55-91-001-\d{2}-\d{4}', folder_name):
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.set_interaction_enabled(False)
                self.merge_thread = MergeAndAnalyzeThread(self.processor, path)
                self.merge_thread.progress_updated.connect(self.progress_bar.setValue)
                self.merge_thread.finished.connect(self.on_merge_and_analyze_finished)
                self.merge_thread.error.connect(self.on_processing_error)
                self.merge_thread.start()
            else:
                QMessageBox.warning(self, "Cảnh báo", "Chỉ hỗ trợ kéo thả thư mục đúng định dạng A55-91-001-xx-xxxx.")
        else:
            QMessageBox.warning(self, "Cảnh báo", "Chỉ hỗ trợ kéo thả file PDF hoặc thư mục đúng định dạng.")

    def select_file(self):
        from PySide6.QtWidgets import QFileDialog
        import os
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.load_pdf(file_path)

    def select_folder(self):
        from PySide6.QtWidgets import QFileDialog
        import os
        import re
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa các file PDF")
        if folder:
            folder_name = os.path.basename(folder)
            if re.match(r'^A55-91-001-\d{2}-\d{4}$', folder_name):
                try:
                    self.progress_bar.setValue(0)
                    self.progress_bar.setVisible(True)
                    self.set_interaction_enabled(False)
                    self.merge_thread = MergeAndAnalyzeThread(self.processor, folder)
                    self.merge_thread.progress_updated.connect(self.progress_bar.setValue)
                    self.merge_thread.finished.connect(self.on_merge_and_analyze_finished)
                    self.merge_thread.error.connect(self.on_processing_error)
                    self.merge_thread.start()
                except Exception as e:
                    QMessageBox.critical(self, "Lỗi", f"Gộp file thất bại: {str(e)}")
            else:
                QMessageBox.warning(self, "Cảnh báo", "Chỉ hỗ trợ chọn thư mục đúng định dạng A55-91-001-xx-xxxx.")

    def load_pdf(self, filepath):
        self.current_file = filepath
        self.save_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        self.drop_area.setText(f"File đã chọn: {os.path.basename(filepath)}")
        self.progress_bar.setValue(0)
        # Clear thumbnails
        for i in reversed(range(self.thumbnails_layout.count())):
            self.thumbnails_layout.itemAt(i).widget().setParent(None)
        # Vô hiệu hóa mọi tương tác khi phân tích
        self.set_interaction_enabled(False)
        self.analyze_pdf()

    def analyze_pdf(self):
        if not self.current_file:
            return

        self.worker = PDFWorker(
            self.processor, self.current_file, analyze_only=True)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_analysis_complete(self, blank_pages):
        self.set_interaction_enabled(True)
        self.save_btn.setEnabled(True)
        self.save_as_btn.setEnabled(True)
        self.show_thumbnails(blank_pages)
        # Hiển thị số lượng trang và số trang trắng trong drop_area
        if self.current_file:
            import fitz
            doc = fitz.open(self.current_file)
            total_pages = len(doc)
            doc.close()
            file_name = os.path.basename(self.current_file)
            self.drop_area.setText(
                f"File đã chọn {file_name}\nTổng: {total_pages} | Trắng {len(blank_pages)} | Còn lại {total_pages - len(blank_pages)}")
        # if blank_pages:
        #     QMessageBox.information(self, "Phân tích hoàn tất",
        #                             f"Đã tìm thấy {len(blank_pages)} trang trắng")
        # else:
        #     QMessageBox.information(self, "Phân tích hoàn tất",
        #                             "Không tìm thấy trang trắng nào")

    def save_pdf(self, save_as_new):
        if not self.current_file:
            return

        self.worker = PDFWorker(self.processor, self.current_file, save_as_new)
        self.worker.finished.connect(self.on_save_complete)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_save_complete(self, result):
        QMessageBox.information(self, "Lưu hoàn tất",
                                f"File đã được lưu tại:\n{result}")

    def on_processing_error(self, error_msg):
        self.set_interaction_enabled(True)
        QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi:\n{error_msg}")

    def resizeEvent(self, event):
        # Khi cửa sổ thay đổi kích thước, tự động vẽ lại thumbnail cho vừa cột
        if hasattr(self, 'last_blank_pages') and self.last_blank_pages is not None:
            self.show_thumbnails(self.last_blank_pages)
        super().resizeEvent(event)

    def show_thumbnails(self, blank_pages):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnails_layout.count())):
            self.thumbnails_layout.itemAt(i).widget().setParent(None)

        if not blank_pages:
            self.last_blank_pages = []
            return
        self.last_blank_pages = blank_pages

        from PySide6.QtGui import QPixmap, QImage
        columns = 6
        spacing = 12  # spacing giữa các container
        # Cách lề ngoài cùng cho layout
        self.thumbnails_layout.setContentsMargins(8, 8, 8, 8)
        self.thumbnails_layout.setHorizontalSpacing(spacing)
        self.thumbnails_layout.setVerticalSpacing(spacing)
        parent_width = self.thumbnails_area.viewport().width()
        col_width = max((parent_width - (columns - 1) *
                        spacing - 2 * 8) // columns, 60)
        for idx, page_info in enumerate(blank_pages):
            page_num = page_info['page_num']
            img_bytes = page_info['thumbnail']
            image = QImage.fromData(img_bytes, 'PNG')
            pixmap = QPixmap.fromImage(image)
            aspect_ratio = pixmap.height() / pixmap.width() if pixmap.width() else 1.5
            thumb_height = int(col_width * aspect_ratio)
            scaled_pixmap = pixmap.scaled(
                col_width - 8, thumb_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb_label = QLabel()
            thumb_label.setPixmap(scaled_pixmap)
            thumb_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
            thumb_label.setStyleSheet("border: none; background: none;")
            # Tạo widget chứa cả ảnh và số trang
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(4, 4, 4, 8)
            vbox.setSpacing(6)
            vbox.addWidget(thumb_label)
            page_label = QLabel(f"Trang {page_num}")
            page_label.setAlignment(Qt.AlignBottom | Qt.AlignCenter)
            page_label.setStyleSheet("border: none; background: transparent;")
            vbox.addWidget(page_label)
            # Thêm style cho container
            container.setStyleSheet("""
                background: #fff;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 0;
                margin: 0;
                color: #000;
            """)
            self.thumbnails_layout.addWidget(
                container, idx // columns, idx % columns)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục để gộp PDF")
        if not folder:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một thư mục hợp lệ.")
            return
        self.processor.input_folder = folder
        # Tạo thread cho việc gộp file
        self.merge_thread = QThread()
        self.processor.moveToThread(self.merge_thread)
        self.merge_thread.started.connect(self.start_merge)
        self.merge_thread.finished.connect(self.merge_thread.deleteLater)
        self.merge_thread.start()

    def start_merge(self):
        folder = self.processor.input_folder
        # Thêm logging thông báo bắt đầu gộp file
        self.processor.logger.info("Đang bắt đầu gộp file PDF...")
        output = self.processor.merge_pdfs(folder)
        if output:
            self.on_merge_complete(output)
        else:
            self.on_processing_error("Gộp file thất bại")

    def on_merge_complete(self, output):
        self.set_interaction_enabled(True)
        QMessageBox.information(self, "Gộp hoàn tất",
                                f"Đã gộp file và lưu tại:\n{output}")

    def on_merge_and_analyze_finished(self, merged_pdf):
        self.set_interaction_enabled(True)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Gộp PDF hoàn tất", f"Đã gộp file PDF: {merged_pdf}")
        self.load_pdf(merged_pdf)

    def set_interaction_enabled(self, enabled):
        self.select_file_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.save_as_btn.setEnabled(enabled)
        self.drop_area.setAcceptDrops(enabled)
        # Đảm bảo không truy cập merge_thread đã bị xóa
        if hasattr(self, 'merge_thread'):
            try:
                if self.merge_thread.isRunning():
                    self.merge_thread.quit()
                    self.merge_thread.wait()
            except RuntimeError:
                # Nếu merge_thread đã bị xóa bởi Qt, bỏ qua lỗi này
                pass
