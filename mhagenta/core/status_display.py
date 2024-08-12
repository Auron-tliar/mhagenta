import tkinter as tk
import tkinter.ttk as ttk


class StatusDisplay:
    def __init__(self):
        self._window = tk.Tk()

    async def open(self):
        self._window.mainloop()

    def close(self):
        self._window.destroy()

    def add_agent(self):
        pass

    def add_log_entry(self):
        pass
