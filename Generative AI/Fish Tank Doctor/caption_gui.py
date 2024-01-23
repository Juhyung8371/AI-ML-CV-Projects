
# GIT fine tune tutorial
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb

# format
# https://huggingface.co/docs/datasets/main/en/image_dataset#image-captioning

# centercrop 224 px defined here
# https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/layers/CLIP/clip.py

import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
import csv


class ImageView(tk.Frame):
    filename = None
    caption = None

    folder = None
    image_path = None
    image_list = []
    image_index = 0
    image_widget = None
    filename_widget = None
    caption_field = None

    metadata_file_path = 'metadata.csv'
    metadata_list = []

    def __init__(self, root):
        tk.Frame.__init__(self, root)

        # create a 2x2 grid
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # filename
        self.filename = tk.StringVar()
        self.filename_widget = tk.Label(self, textvariable=self.filename)
        self.filename_widget.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.W + tk.E)
        # image
        self.image_widget = tk.Label(self)
        self.image_widget.grid(row=1, column=0, sticky=tk.W + tk.S + tk.N)
        # caption field
        self.caption_field = tk.Text(self, wrap="word", width=50, font=("Arial", 16))
        self.caption_field.grid(row=1, column=1, sticky=tk.E + tk.S + tk.N)

    def open_folder(self):
        self.folder = Path(filedialog.askdirectory())
        if self.folder is None:
            return
        self.image_list.clear()
        for file in os.listdir(self.folder):
            f = Path(file)
            if f.suffix == '.jpg':
                self.image_list.append(f)

        self.image_list.sort()
        self.load_image()
        load_csv = self.load_csv()

        if load_csv:
            self.load_label_txt()

    def load_csv(self):

        print('----------------------------------')

        path = Path(self.metadata_file_path)

        num_images = len(self.image_list)

        if path.exists():

            print(f'The csv file exists. Loading {path.as_posix()}')

            with open(self.metadata_file_path, mode='r') as csv_file:

                # the first row is formatting establishment, not data
                skipped_first_row = False

                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:

                    if not skipped_first_row:
                        skipped_first_row = True
                        continue

                    self.metadata_list.append([row[0], row[1]])

            print('Loaded csv to a list.')
            print(f'Number of rows: {len(self.metadata_list)}')

            print('----------------------------------')
            return True
        else:
            print(f'csv file does not exist at {path.as_posix()}')

            # initialize with empty ones
            for row in range(num_images):
                self.metadata_list.append([f'img ({row+1}).jpg', ''])

            print('Initialized an empty list.')
            print(f'Number of rows: {len(self.metadata_list)}')

            print('----------------------------------')
            return False

    def load_image(self):
        self.image_path = self.folder / f'img ({self.image_index+1}).jpg'
        self.filename.set(f'img ({self.image_index+1}).jpg')
        img = Image.open(self.image_path)

        width, height = img.size

        new_width = 480
        new_height = int(new_width * height / width)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        if img.width > root.winfo_width() or img.height > root.winfo_height():
            img.thumbnail((480, 480))

        img = ImageTk.PhotoImage(img)
        self.image_widget.configure(image=img)
        self.image_widget.image = img

    def save_csv(self):

        path = Path(self.metadata_file_path)

        print('----------------------------------')
        print(f'Saving the list to a csv at {path.as_posix()}')
        print(f'Last img worked on: img ({self.image_index+1}).jpg')

        with open(self.metadata_file_path, mode='w', newline='') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')

            # establish the file format in the first line
            employee_writer.writerow(['file_name', 'text'])

            for row in self.metadata_list:
                title = row[0]
                text = row[1]

                employee_writer.writerow([title, text])

        print('Saved')
        print('----------------------------------')

    def write_label_txt(self):
        # read from start to second last character
        label_text = self.caption_field.get("1.0", "end-1c")
        label_text = label_text.replace('\r', '').replace('\n', '').strip()

        label_title = f'img ({self.image_index+1}).jpg'

        print(f'Write at {label_title}, Text: {label_text[:30]} (...)')
        self.metadata_list[self.image_index] = [label_title, label_text]

    def load_label_txt(self):
        # load label text file for the new image
        self.caption_field.delete(1.0, tk.END)

        if 0 <= self.image_index < len(self.metadata_list):
            text = self.metadata_list[self.image_index][1]
            self.caption_field.insert(tk.END, text)

    def go_to_image(self, index):
        self.write_label_txt()
        self.image_index = index
        self.load_label_txt()
        self.load_image()

    def next_image(self):
        self.save_csv()
        self.write_label_txt()
        self.image_index += 1
        if self.image_index >= len(self.image_list):
            self.image_index = 0
        self.load_label_txt()
        self.load_image()

    def prev_image(self):
        self.save_csv()
        self.write_label_txt()
        self.image_index -= 1
        if self.image_index < 0:
            self.image_index = len(self.image_list) - 1
        self.load_label_txt()
        self.load_image()

    # move current image to a "_deleted" folder
    def delete_image(self):
        if len(self.image_list) == 0:
            return
        cur_image_name = self.image_list[self.image_index]
        cur_image_path = self.folder / cur_image_name
        self.next_image()
        deleted_folder = Path(self.folder / '_deleted')
        if not deleted_folder.exists():
            deleted_folder.mkdir()
        os.rename(cur_image_path, deleted_folder / cur_image_name)
        # move the corresponding text file to the deleted folder
        txt_file_name = cur_image_name.stem + '.txt'
        label_txt_file = self.folder / txt_file_name
        if label_txt_file.exists():
            os.rename(label_txt_file, deleted_folder / txt_file_name)


# [control + o] to open a folder of images.
# [page down] and [page up] to go to next and previous images. Hold shift to skip 10 images.
# [escape] to exit the app.

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1200x800')
    root.title('Image Captions')

    root.bind('<Control-o>', lambda e: view.open_folder())
    root.bind('<Control-s>', lambda e: view.save_csv())
    root.bind('<Escape>', lambda e: root.destroy())
    root.bind('<Prior>', lambda e: view.prev_image())
    root.bind('<Next>', lambda e: view.next_image())
    root.bind('<Shift-Prior>', lambda e: view.go_to_image(view.image_index - 10))
    root.bind('<Shift-Next>', lambda e: view.go_to_image(view.image_index + 10))
    # root.bind('<Shift-Home>', lambda e: view.go_to_image(0))
    # root.bind('<Shift-End>', lambda e: view.go_to_image(len(view.image_list) - 1))
    # root.bind('<Shift-Delete>', lambda e: view.delete_image())

    view = ImageView(root)
    view.pack(side="top", fill="both", expand=True)
    root.mainloop()
