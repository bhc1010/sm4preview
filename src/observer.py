import time, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from sm4preview import SM4File

class Watcher:
    DIRECTORY_TO_WATCH = r"C:\Users\spmuser\OneDrive - USNH\Hollen Lab\Data\LEWIS"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            root, ext = os.path.splitext(event.src_path)
            for banned in ['Memscope', '-FFT', '-Scope', '-IZ']:
                if banned in root:
                    return
            if ext == '.sm4':
                print(f'Generating preview for {event.src_path}')
                time.sleep(1)
                try:
                    sm4 = SM4File(event.src_path)
                    sm4.generate_preview()
                    del sm4
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    w = Watcher()
    w.run()