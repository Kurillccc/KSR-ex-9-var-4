import matplotlib.pyplot as plt
import tkinter as tk
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('Agg')

from modules.RungeKuttSystem import *

from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_plot():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    U1_0 = float(u1_0_entry.get())
    U2_0 = float(u2_0_entry.get())
    step_type = step_type_var.get()
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    m = float(m_entry.get())

    func1 = RungeKuttaSystem(h0, x_0, U1_0, U2_0, maxCount, epsilonG, a1, a3, m)
    data = np.array([func1.variableSteps(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func1.fixecStep(xMax)])
    x, u1, u2 = data.T
    V2 = func1.V2
    OLP = func1.OLP
    Hi = func1.Hi
    C1 = func1.C1
    C2 = func1.C2

    tree.delete(*tree.get_children())

    Data = []
    for i in range(1, len(x)):
        Data.append((i, x[i], [u1[i], u2[i]], V2[i - 1] if i - 1 < len(V2) else "",
                     [V2[i - 1][0] - u1[i], V2[i - 1][1] - u2[i]] if i - 1 < len(V2) else "",
                     OLP[i - 1] if i - 1 < len(OLP) else "", Hi[i - 1] if i - 1 < len(Hi) else "",
                     C1[i - 1] if i - 1 < len(C1) else "",
                     C2[i - 1] if i - 1 < len(C2) else "", "", ""))

    for inf in Data:
        tree.insert("", "end", values=inf)

    number_of_iterations = len(x) - 1
    difference = xMax - x[number_of_iterations]
    maxOLP = max(OLP) if len(OLP) != 0 else 0
    C_1 = max(C1) if len(C1) != 0 else 0
    C_2 = max(C2) if len(C2) != 0 else 0
    max_h = max(Hi) if len(Hi) != 0 else h0
    min_h = min(Hi) if len(Hi) != 0 else h0
    max_h_index = Hi.index(max_h) + 1 if len(Hi) != 0 else 0
    min_h_index = Hi.index(min_h) + 1 if len(Hi) != 0 else 0
    max_h_x = x[max_h_index] if len(Hi) != 0 else 0
    min_h_x = x[min_h_index] if len(Hi) != 0 else 0

    results_window = tk.Toplevel(root)
    results_window.title("Результаты")

    tk.Label(results_window, text="Количество итераций:").grid(row=0, column=0)
    tk.Label(results_window, text=number_of_iterations).grid(row=0, column=1)

    tk.Label(results_window, text="Разница между правой границей и последним вычисленным значением:").grid(row=1,
                                                                                                           column=0)
    tk.Label(results_window, text=difference).grid(row=1, column=1)

    tk.Label(results_window, text="Максимальное значение OLP:").grid(row=2, column=0)
    tk.Label(results_window, text=maxOLP).grid(row=2, column=1)

    tk.Label(results_window, text="Количество делений:").grid(row=3, column=0)
    tk.Label(results_window, text=C_1).grid(row=3, column=1)

    tk.Label(results_window, text="Количество удвоений:").grid(row=4, column=0)
    tk.Label(results_window, text=C_2).grid(row=4, column=1)

    tk.Label(results_window, text="Максимальное значение Hi:").grid(row=5, column=0)
    tk.Label(results_window, text=max_h).grid(row=5, column=1)

    tk.Label(results_window, text="Минимальное значение Hi:").grid(row=6, column=0)
    tk.Label(results_window, text=min_h).grid(row=6, column=1)

    tk.Label(results_window, text="Значение x для максимального Hi:").grid(row=9, column=0)
    tk.Label(results_window, text=max_h_x).grid(row=9, column=1)

    tk.Label(results_window, text="Значение x для минимального Hi:").grid(row=10, column=0)
    tk.Label(results_window, text=min_h_x).grid(row=10, column=1)

    fig, axarr = plt.subplots(3, sharex=True, figsize=(8, 10))


    axarr[0].plot(x, u1, label='U1(x)')
    axarr[0].set_ylabel('U1')
    axarr[0].legend()

    axarr[1].plot(x, u2, label='U2(x)')
    axarr[1].set_ylabel('U2')
    axarr[1].legend()

    axarr[2].plot(u1, u2, label='U2(U1)')
    axarr[2].set_xlabel('U1')
    axarr[2].set_ylabel('U2')
    axarr[2].legend()

    plt.xlabel("x")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Численные методы задание 9 вариант 4")

    frame = ttk.Frame(root)
    frame.pack(padx=10, pady=10)

    u1_0_label = ttk.Label(frame, text="U1_0:")
    u1_0_label.grid(row=0, column=2)
    u1_0_entry = ttk.Entry(frame)
    u1_0_entry.grid(row=0, column=3)
    u1_0_entry.insert(0,"1")

    u2_0_label = ttk.Label(frame, text="U2_0:")
    u2_0_label.grid(row=1, column=2)
    u2_0_entry = ttk.Entry(frame)
    u2_0_entry.grid(row=1, column=3)
    u2_0_entry.insert(0,"1")


    maxCount_label = ttk.Label(frame, text="Максимальное количество итераций:")
    maxCount_label.grid(row=0, column=0)
    maxCount_entry = ttk.Entry(frame)
    maxCount_entry.grid(row=0, column=1)
    maxCount_entry.insert(0, "10000")

    maxError_label = ttk.Label(frame, text="Максимальная ошибка:")
    maxError_label.grid(row=1, column=0)
    maxError_entry = ttk.Entry(frame)
    maxError_entry.grid(row=1, column=1)
    maxError_entry.insert(0, "0.0001")

    h0_label = ttk.Label(frame, text="Начальный шаг:")
    h0_label.grid(row=2, column=0)
    h0_entry = ttk.Entry(frame)
    h0_entry.grid(row=2, column=1)
    h0_entry.insert(0, "0.01")

    xMax_label = ttk.Label(frame, text="Правая граница:")
    xMax_label.grid(row=3, column=0)
    xMax_entry = ttk.Entry(frame)
    xMax_entry.grid(row=3, column=1)
    xMax_entry.insert(0, "1.69")

    x0_label = ttk.Label(frame, text="x0:")
    x0_label.grid(row=2, column=2)
    x0_entry = ttk.Entry(frame)
    x0_entry.grid(row=2, column=3)
    x0_entry.insert(0, "1")

    u0_label = ttk.Label(frame, text="u0:")
    u0_label.grid(row=3, column=2)
    u0_entry = ttk.Entry(frame)
    u0_entry.grid(row=3, column=3)
    u0_entry.insert(0, "1")

    epsilonG_label = ttk.Label(frame, text="Епселон граничный:")
    epsilonG_label.grid(row=6, column=0)
    epsilonG_entry = ttk.Entry(frame)
    epsilonG_entry.grid(row=6, column=1)
    epsilonG_entry.insert(0, "0.001")

    task_var = tk.StringVar()

    step_type_label = ttk.Label(frame, text="Выберите шаг:")
    step_type_label.grid(row=0, column=7)
    step_type_var = tk.StringVar()
    step_type_var.set("Фиксированный")
    step_type_option = ttk.OptionMenu(frame, step_type_var, "Фиксированный","Фиксированный", "Переменный")
    step_type_option.grid(row=0, column=8)

    a1_label = ttk.Label(frame, text="a1:")
    a1_label.grid(row=1, column=7)
    a1_entry = ttk.Entry(frame)
    a1_entry.grid(row=1, column=8)
    a1_entry.insert(0, "1")

    a3_label = ttk.Label(frame, text="a3:")
    a3_label.grid(row=2, column=7)
    a3_entry = ttk.Entry(frame)
    a3_entry.grid(row=2, column=8)
    a3_entry.insert(0, "1")

    m_label = ttk.Label(frame, text="m:")
    m_label.grid(row=3, column=7)
    m_entry = ttk.Entry(frame)
    m_entry.grid(row=3, column=8)
    m_entry.insert(0, "1")

    update_button = ttk.Button(frame, text="Обновить", command=update_plot)
    update_button.grid(row=11, columnspan=2)


    columns = ("i", "xi", "vi", "v2i", "vi-v2i", "OLP", "hi", "C1", "C2", "ui", "ui-vi")
    tree = ttk.Treeview(columns=columns, show="headings")

    vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    vsb.pack(side="right", fill="y")
    tree.configure(yscrollcommand=vsb.set)

    tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    for col in columns:
        tree.heading(col, text=col)

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    update_plot()

    root.mainloop()