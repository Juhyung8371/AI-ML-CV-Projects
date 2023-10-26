import threading
from time import strftime
from tkinter import *


class GUI:
    """ GUI for mouse testing purposes."""

    def __init__(self):
        # initialize
        self.root = None
        self.stopped = False
        self.t = threading.Thread(target=self.__main)

    def __main(self):
        self.root = Tk()
        self.root.geometry("535x500")
        self.root.title("Mouse Testing - Juhyung")
        self.root.resizable(False, False)  # not resizable

        # buttons #########################################

        # padding for widgets inside btn_frame
        # padx goes to radiobuttons and pady goes to every labels
        btn_frame_padx = 15
        btn_frame_pady = 10

        # the frame on the left side
        btn_frame = Frame(self.root, borderwidth=2, relief=RIDGE)
        # the frame should span 2 rows since it has many widgets
        btn_frame.grid(row=0, column=0, rowspan=2, sticky="NS")

        # button label 1
        btn_label1 = Label(btn_frame, text="Buttons")
        btn_label1.grid(row=0, column=0, pady=btn_frame_pady, sticky="W")

        # button
        def on_click_single_click():
            set_text_input("Single click")

        def on_click_double_click(event):
            set_text_input("Double click")

        single_click_btn = Button(btn_frame, text="Single click", command=on_click_single_click)
        single_click_btn.grid(row=1, column=0)

        double_click_btn = Button(btn_frame, text="Double click")
        double_click_btn.grid(row=1, column=1)
        double_click_btn.bind('<Double-Button-1>', on_click_double_click)

        # button label 2
        btn_label2 = Label(btn_frame, text="Radio Buttons")
        btn_label2.grid(row=2, column=0, pady=btn_frame_pady, sticky="W")

        # radio button #########################################

        def radio_select():
            selection = "Radiobutton: " + str(radio_var.get())
            set_text_input(selection)

        radio_var = IntVar()

        R1 = Radiobutton(btn_frame, text="Option 1", variable=radio_var, value=1, command=radio_select)
        R1.grid(row=3, column=0, padx=btn_frame_padx)

        R2 = Radiobutton(btn_frame, text="Option 2", variable=radio_var, value=2, command=radio_select)
        R2.grid(row=3, column=1, padx=btn_frame_padx)

        R3 = Radiobutton(btn_frame, text="Option 3", variable=radio_var, value=3, command=radio_select)
        R3.grid(row=3, column=2, padx=btn_frame_padx)

        # checkbox #########################################

        btn_label3 = Label(btn_frame, text="Checkbox")
        btn_label3.grid(row=4, column=0, pady=btn_frame_pady, sticky="W")

        checkbox_var1 = StringVar()
        checkbox_var2 = StringVar()
        checkbox_var3 = StringVar()

        def checkbox_select():
            selection = ("Checkbox: " + checkbox_var1.get()
                         + checkbox_var2.get()
                         + checkbox_var3.get())

            set_text_input(selection)

        checkbox1 = Checkbutton(btn_frame, text="Option 1", variable=checkbox_var1,
                                onvalue="V", offvalue="O", command=checkbox_select)
        checkbox1.deselect()
        checkbox1.grid(row=5, column=0)

        checkbox2 = Checkbutton(btn_frame, text="Option 2", variable=checkbox_var2,
                                onvalue="V", offvalue="O", command=checkbox_select)
        checkbox2.deselect()
        checkbox2.grid(row=5, column=1)

        checkbox3 = Checkbutton(btn_frame, text="Option 3", variable=checkbox_var3,
                                onvalue="V", offvalue="O", command=checkbox_select)
        checkbox3.deselect()
        checkbox3.grid(row=5, column=2)

        # Text entry #########################################

        btn_label4 = Label(btn_frame, text="Text Entry")
        btn_label4.grid(row=6, column=0, pady=btn_frame_pady)

        # input field
        input1 = Entry(btn_frame, width=10)
        input1.insert(0, "Text Entry")
        input1.grid(row=7, column=0)

        # Option menu #########################################

        btn_label5 = Label(btn_frame, text="Option Menu")
        btn_label5.grid(row=6, column=1, pady=btn_frame_pady)

        options = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

        option_var = StringVar()
        option_var.set(options[0])  # default value
        option_menu = OptionMenu(btn_frame, option_var, *options)
        option_menu.grid(row=7, column=1)

        # spinbox #########################################

        btn_label6 = Label(btn_frame, text="Spinbox")
        btn_label6.grid(row=6, column=2, pady=btn_frame_pady)

        current_value = IntVar(value=0)
        # wrap allows you to from 0 to 10 backward too
        spin_box = Spinbox(btn_frame, from_=0, to=10, width=10,
                           textvariable=current_value, wrap=True)
        spin_box.grid(row=7, column=2)

        # Sliders #########################################

        h_slider = Scale(btn_frame, from_=0, to=300, length=150, orient=HORIZONTAL)
        h_slider.grid(row=10, column=0, columnspan=2)
        v_slider = Scale(btn_frame, from_=0, to=300, length=150, orient=VERTICAL)
        v_slider.grid(row=10, column=2, rowspan=2)

        # canvas #########################################

        canvas_frame = Frame(self.root)
        canvas_frame.grid(row=0, column=1)

        # button label 1
        canvas_label = Label(canvas_frame, text="Canvas")
        canvas_label.pack()

        def draw(event):
            # coord
            x1, y1 = event.x, event.y
            # draw a circle in each spot
            canvas.create_oval(x1 - 2, y1 - 2, x1 + 2, y1 + 2, fill="black", outline="black")

        canvas = Canvas(canvas_frame, height=200, width=200, bg="white", bd=2, relief=RIDGE)
        # hold and drag motion detection
        canvas.bind("<B1-Motion>", draw)
        canvas.pack(padx=10)

        # canvas reset button
        def on_click_reset_canvas():
            canvas.delete("all")
            set_text_input("Canvas cleared")

        canvas_reset_btn = Button(canvas_frame, width=25,
                                  text="Reset Canvas", command=on_click_reset_canvas)
        canvas_reset_btn.pack(pady=5)

        # prompt textfield ###############################

        def set_text_input(text):
            text_prompt.insert("end", text + " " + strftime('%H:%M:%S %p') + "\n")
            text_prompt.see("end")  # scroll to the end

        textContainer = Frame(self.root, borderwidth=1, relief="sunken")
        textContainer.grid(row=1, column=1)

        # configure the scroll commands
        text_prompt = Text(textContainer, width=24, height=13, wrap="none", borderwidth=0)
        textVsb = Scrollbar(textContainer, orient="vertical", command=text_prompt.yview)
        textHsb = Scrollbar(textContainer, orient="horizontal", command=text_prompt.xview)
        text_prompt.configure(yscrollcommand=textVsb.set, xscrollcommand=textHsb.set)

        # let them stretch
        text_prompt.grid(row=0, column=0, sticky="nsew")
        textVsb.grid(row=0, column=1, sticky="ns")
        textHsb.grid(row=1, column=0, sticky="ew")

        # match the weight so the widgets will resize proportionally in 1:1 ratio
        textContainer.grid_rowconfigure(0, weight=1)
        textContainer.grid_columnconfigure(0, weight=1)

        # display the things
        self.root.mainloop()
        self.stopped = True

    def show(self):
        self.t.start()

    def close(self):
        self.root.destroy()
