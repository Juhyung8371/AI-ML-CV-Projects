# GIT fine tune tutorial
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb

# format
# https://huggingface.co/docs/datasets/main/en/image_dataset#image-captioning

# centercrop 224 px defined here
# https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/layers/CLIP/clip.py

# CURRENT 509


import csv
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from PIL import Image, ImageTk
import image_utils


class CaptionGUI(tk.Frame):
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

        # create a row x col = 21x5 grid
        self.grid_columnconfigure(0, weight=2)

        # filename
        self.filename = tk.StringVar()
        self.filename_widget = tk.Label(self, textvariable=self.filename)
        self.filename_widget.grid(row=0, column=0)

        # image
        self.image_widget = tk.Label(self)
        self.image_widget.grid(row=1, column=0, rowspan=22, padx=(5, 5), pady=(5, 5), sticky=tk.S)

        self.caption_text_fields = []
        self.caption_delete_buttons = []
        self.caption_add_buttons = []

        # add caption fields
        num_caption_fields = 10
        caption_index = -1
        for col in range(1, 5, +2):  # there are 4 columns but 2 are done at once, so iterate 2 times
            for row in range(0, num_caption_fields):
                # caption field
                caption_field = tk.Text(self, wrap="word", height=1, width=30, font=("Arial", 16))
                caption_field.grid(row=row, column=col, padx=(5, 0), pady=(2, 2))
                self.caption_text_fields.append(caption_field)
                caption_index += 1

                # caption delete button
                def delete_caption(index=caption_index):
                    self.caption_text_fields[index].delete(1.0, "end")

                caption_delete_button = tk.Button(self, text='X', width=5, command=delete_caption)
                caption_delete_button.grid(row=row, column=col + 1)
                self.caption_delete_buttons.append(caption_delete_button)

        # add 'add front or back' toggle button
        # highlight active text field
        def toggle(event):
            if toggle_btn["text"] == "Add caption BACK":
                toggle_btn["text"] = "Add caption FRONT"
                self.add_caption_back = False

                self.caption_text_fields[-1].config(background="white")
                self.caption_text_fields[0].config(background="green")

            elif toggle_btn["text"] == "Add caption FRONT":
                toggle_btn["text"] = "Add caption BACK"
                self.add_caption_back = True

                self.caption_text_fields[0].config(background="white")
                self.caption_text_fields[-1].config(background="green")

        self.add_caption_back = True
        toggle_btn = tk.Button(self, text='Add caption BACK')
        toggle_btn.grid(row=num_caption_fields, column=1, pady=(5, 5), columnspan=4, sticky=tk.W + tk.E)
        toggle_btn.bind("<Button-1>", toggle)

        # highlight active text field
        self.caption_text_fields[-1].config(background="green")

        # quick caption buttons frame
        quick_caption_frame = tk.Frame(self)
        quick_caption_frame.grid(row=num_caption_fields + 1, column=1, columnspan=4, rowspan=15,
                                 sticky=tk.W + tk.E + tk.S + tk.N)

        # list of captions by category
        quick_captions = {
            'layout': ['natural layout', 'spacious layout', 'balanced hardscape', 'plants thriving',
                       'nutrient-rich substrate', 'clean bottom', 'dirty substrate',
                       'consider live plants', 'trim plant', 'artificial decorations', 'consider natural layout',
                       'cluttered layout', 'organize the layout', ],
            'fish': ['healthy fish', 'adequate fish stocking', 'overstocked', 'possibly overcrowded',
                     'check fish compatibility', 'monitor fish health', ],
            'tank': ['small tank', 'larger tank needed', 'consider larger tank', 'adequate tank size', ],
            'water': ['clean water', 'cloudy water', 'clear but brown water',
                      'dirty water', 'water change regularly', 'raise water level', 'check water parameters', ],
            'equipments': ['add filter', 'add heater', 'check filter', 'balance temperature', 'good aeration',
                           'well-hidden equipments', 'too bright', 'too dark', 'uneven light', 'balance lighting', ],
            'algae': ['algae issue', 'avoid overfeeding', 'wipe glass', ],
            'new': ['newly set up', 'cycle the water', 'check for leak', ],
            'misc': ['not a full aquarium image', 'an empty tank', 'not an aquarium image',
                     'provide different image', ],
        }

        for y, (key, caption_list) in enumerate(quick_captions.items()):

            quick_caption_inner_frame = tk.Frame(quick_caption_frame)
            quick_caption_inner_frame.grid(row=0, column=y, rowspan=15, sticky=tk.W + tk.E + tk.S + tk.N)

            key_label = tk.Label(quick_caption_inner_frame, text=key)
            key_label.grid(row=0, column=0, pady=(5, 5), sticky=tk.W + tk.E)

            for i in range(0, len(caption_list)):

                # initialize button first to give the instance to the command function
                temp_quick_cap_btn = tk.Button(quick_caption_inner_frame, text=caption_list[i])

                def apply_quick_caption(widget=temp_quick_cap_btn):

                    text_to_add = widget['text']

                    caption_fields_count = len(self.caption_text_fields)

                    # assume add back, so push from front
                    iteration_range = range(caption_fields_count)  # order to iterate
                    first_caption_index = 0  # index of text field to skip because the first one can't be pushed
                    insert_caption_index = caption_fields_count - 1  # index of text field add new caption
                    iteration_factor = -1  # -1 for backward iteration, +1 otherwise

                    # otherwise, update values for the front-adding case
                    if not self.add_caption_back:
                        iteration_range = reversed(iteration_range)
                        first_caption_index = caption_fields_count - 1
                        insert_caption_index = 0
                        iteration_factor = 1

                    # push texts forward or backward to make room
                    for n in iteration_range:

                        current_text_field = self.caption_text_fields[n]

                        if n != first_caption_index:

                            next_text_field = self.caption_text_fields[n + iteration_factor]
                            current_text = current_text_field.get("1.0", "end-1c")
                            next_text = next_text_field.get("1.0", "end-1c")

                            if next_text == '' and len(current_text) > 0:
                                current_text_field.delete(1.0, "end")
                                next_text_field.delete(1.0, "end")
                                next_text_field.insert(tk.END, current_text)

                    # add text in the last text field if possible
                    front_or_back_text_field = self.caption_text_fields[insert_caption_index]
                    last_text = front_or_back_text_field.get("1.0", "end-1c")
                    if last_text == '':
                        front_or_back_text_field.insert(tk.END, text_to_add)

                # now apply the command
                temp_quick_cap_btn.configure(command=apply_quick_caption)
                temp_quick_cap_btn.grid(row=i + 1, column=0, padx=(2, 2), pady=(1, 0), sticky=tk.W + tk.E)

    def open_folder(self):
        self.folder = Path(filedialog.askdirectory())

        if self.folder is None:
            return

        self.image_list.clear()

        self.image_list = image_utils.get_image_names(str(self.folder), return_full_path=False)

        self.load_image()

        load_csv = self.load_csv()

        if load_csv:
            self.load_label_txt()

    def filter_caption(self, caption):
        """
        Filter the caption to maintain the consistent format and save tokens
        :param caption:
        :return:
        """
        return caption.lower().replace('.', '').replace('\n', ' ').replace('potential improvements:', '')

    def load_csv(self):

        print('----------------------------------')

        path = Path(self.metadata_file_path)

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

                    self.metadata_list.append([row[0], self.filter_caption(row[1])])

                current_len = len(self.metadata_list)

                print(f'Loaded {current_len} from csv file.')

                # initialize empty ones
                for i, image_name in enumerate(self.image_list):
                    if i > current_len:
                        self.metadata_list.append([image_name, ''])

            print(f'Number of images: {len(self.metadata_list)}')

            print('----------------------------------')
            return True
        else:
            print(f'csv file does not exist at {path.as_posix()}')

            # initialize with empty ones
            for image_name in self.image_list:
                self.metadata_list.append([image_name, ''])

            print('Initialized an empty list.')
            print(f'Number of images: {len(self.metadata_list)}')

            print('----------------------------------')
            return False

    def load_image(self):
        self.image_path = self.folder / f'img ({self.image_index + 1}).jpg'
        self.filename.set(f'img ({self.image_index + 1}).jpg')
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
        print(f'Last img worked on: img ({self.image_index + 1}).jpg')

        with open(self.metadata_file_path, mode='w', newline='') as employee_file:
            the_writer = csv.writer(employee_file, delimiter=',')

            # establish the file format in the first line
            the_writer.writerow(['file_name', 'text'])

            for row in self.metadata_list:
                title = row[0]
                text = self.filter_caption(row[1])

                the_writer.writerow([title, text])

        print('Saved')
        print('----------------------------------')

    def write_label_txt(self):

        split_list = []

        for text_field in self.caption_text_fields:
            the_text = text_field.get("1.0", "end-1c")
            if the_text != '':
                split_list.append(the_text)

        label_text = ', '.join(split_list)
        label_text = label_text.replace('\r', '').replace('\n', '').strip()

        label_title = f'img ({self.image_index + 1}).jpg'

        print(f'Write at {label_title}, Text: {label_text[:30]} (...)')
        self.metadata_list[self.image_index] = [label_title, label_text]

    def load_label_txt(self):
        # load label text file for the new image

        if 0 <= self.image_index < len(self.metadata_list):
            raw_text = self.metadata_list[self.image_index][1]

            # there are 20 text fields.
            # Any remaining un-split list will be all shoved in the last text field
            split_list = raw_text.split(', ', maxsplit=19)
            split_number = len(split_list)

            for i in range(len(self.caption_text_fields)):

                text_field = self.caption_text_fields[i]
                text_field.delete(1.0, "end")

                if i < split_number:
                    text_field.insert(tk.END, split_list[i])

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
    root.geometry('1500x800')
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

    view = CaptionGUI(root)
    view.pack(side="top", fill="both", expand=True)
    root.mainloop()
