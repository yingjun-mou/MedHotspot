import tkinter as tk
import os


def show_entry_fields():
    os.system('python a.py %s'%e1.get())
    #print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

master = tk.Tk()
master.title("Welcome to MedHotspots app")
master.geometry('350x200')
master.configure(bg='lightblue')

tk.Label(master, bg='white',
         text="Please Enter a Search Term").grid(row=0)
"""tk.Label(master,
         text="Last Name").grid(row=1)"""

e1 = tk.Entry(master)
#e2 = tk.Entry(master)

e1.grid(row=0, column=1)
#e2.grid(row=1, column=1)

tk.Button(master,
          text='Quit', 
          command=master.quit).grid(row=3,
                                    column=0,
                                    sticky=tk.W,
                                    pady=4)
tk.Button(master,
          text='Search', command=show_entry_fields).grid(row=3,
                                                       column=1,
                                                       sticky=tk.W,
                                                       pady=4)



tk.mainloop()
