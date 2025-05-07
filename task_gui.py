import tkinter as tk
from tkinter import ttk, messagebox
import sys
import io
from run_tasks import registered_tasks, run_task

class StdoutRedirector(io.TextIOBase):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass  # für Kompatibilität mit sys.stdout

class TaskRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Task Runner mit Queue und Log")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        self.selected_task = tk.StringVar()
        self.arg_entries = []
        self.task_queue = []
        self.recent_tasks = []

        title_label = ttk.Label(root, text="Task-Auswahl", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(10, 5))

        self.task_dropdown = ttk.Combobox(root, values=list(registered_tasks.keys()), textvariable=self.selected_task, state="readonly", font=("Helvetica", 11))
        self.task_dropdown.pack(fill="x", padx=20, pady=5)
        self.task_dropdown.bind("<<ComboboxSelected>>", self.on_task_selected)

        self.desc_label = tk.Label(root, text="", font=("Helvetica", 10, "italic"), wraplength=760, justify="left")
        self.desc_label.pack(padx=20, pady=(0, 10), anchor="w")

        self.arg_frame = ttk.LabelFrame(root, text="Parameter", padding=(10, 10))
        self.arg_frame.pack(fill="x", padx=20, pady=10)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)

        ttk.Button(btn_frame, text="➕ Zur Queue hinzufügen", command=self.add_to_queue).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="🚀 Queue ausführen", command=self.run_queue).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="🗑️ Queue leeren", command=self.clear_queue).pack(side="left", padx=10)

        queue_label = ttk.Label(root, text="Geplante Tasks:", font=("Helvetica", 11, "bold"))
        queue_label.pack(anchor="w", padx=20)

        self.queue_listbox = tk.Listbox(root, height=5)
        self.queue_listbox.pack(fill="x", padx=20, pady=(0, 10))

        log_label = ttk.Label(root, text="📜 Log-Ausgabe:", font=("Helvetica", 11, "bold"))
        log_label.pack(anchor="w", padx=20)

        log_frame = ttk.Frame(root)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", height=15)
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # stdout umleiten
        sys.stdout = StdoutRedirector(self.log_text)
        sys.stderr = StdoutRedirector(self.log_text)

    def on_task_selected(self, event=None):
        task_name = self.selected_task.get()
        if not task_name:
            return

        # Als zuletzt verwendet markieren
        if task_name in self.recent_tasks:
            self.recent_tasks.remove(task_name)
        self.recent_tasks.insert(0, task_name)
        self.recent_tasks = self.recent_tasks[:3]  # Max. 3 merken

        self.update_dropdown_items()

        # Eingabefelder aktualisieren wie gehabt
        for widget in self.arg_frame.winfo_children():
            widget.destroy()
        self.arg_entries.clear()

        func = registered_tasks[task_name]
        doc = func.__doc__.strip() if func.__doc__ else "Keine Beschreibung vorhanden."
        self.desc_label.config(text=doc)

        args = func.__code__.co_varnames
        if "line_name" in args:
            self._add_arg_field("line_name")
        if "cont_name" in args:
            self._add_arg_field("cont_name")
        if "cut_out_range" in args:
            self._add_arg_field("cut_out_range (z. B. -1000,1000)")
        if "output_dir" in args:
            self._add_arg_field("output_dir (optional)")
        if "plot" in args:
            self._add_arg_field("plot (True/False)")

    def update_dropdown_items(self):
        # Erst die letzten, dann den Rest
        all_tasks = list(registered_tasks.keys())
        remaining = [t for t in all_tasks if t not in self.recent_tasks]
        combined = [f"🔁 {t}" for t in self.recent_tasks] + remaining
        self.task_dropdown['values'] = combined

    def _add_arg_field(self, label_text):
        frame = ttk.Frame(self.arg_frame)
        frame.pack(fill="x", pady=4)
        label = ttk.Label(frame, text=label_text + ": ", width=20, anchor="w")
        label.pack(side="left")
        entry = ttk.Entry(frame)
        entry.pack(side="right", expand=True, fill="x")
        self.arg_entries.append((label_text.split()[0], entry))

    def add_to_queue(self):
        task_name = self.selected_task.get().replace("🔁 ", "")
        if not task_name:
            messagebox.showwarning("Hinweis", "Bitte zuerst einen Task auswählen.")
            return

        args = []
        for name, entry in self.arg_entries:
            val = entry.get().strip()
            if val:
                args.append(val)

        full_command = f"{task_name}::{'::'.join(args)}" if args else task_name
        self.task_queue.append(full_command)
        self.queue_listbox.insert(tk.END, full_command)

    def run_queue(self):
        if not self.task_queue:
            messagebox.showinfo("Leere Queue", "Keine Tasks in der Queue.")
            return

        try:
            run_task(self.task_queue)
            print("\n✅ Alle Tasks erfolgreich ausgeführt.\n")
        except Exception as e:
            print(f"\n❌ Fehler beim Ausführen: {e}\n")
        self.clear_queue()

    def clear_queue(self):
        self.task_queue.clear()
        self.queue_listbox.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = TaskRunnerApp(root)
    root.mainloop()
