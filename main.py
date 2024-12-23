import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules.RKSystemVar3 import *
from modules.RungeKuttSystem import *

matplotlib.use('TkAgg')


def C(x0,y0):
    return y0 / (np.exp(x0))

# --------------------------Вывод таблицы и отрисовка численного решения--------------------------
def update_plot():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    step_type = step_type_var.get()
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    m = float(m_entry.get())

    func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3, m)
    data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array([func.fixedStep(xMax)])
    x,y = data.T
    V2 = func.V2
    OLP = func.OLP
    Hi = func.Hi
    C1 = func.C1
    C2 = func.C2

    tree.delete(*tree.get_children())

    Data = []
    if step_type == "Переменный":
        Data.append((0, x_0, y_0, "", "",
                     "", h0,
                     "",
                     ""))
    else:
        Data.append((0, x_0, y_0, "", "",
                     "", "",
                     "",
                     ""))
    for i in range(1, len(x)):
        Data.append((i, x[i], y[i], V2[i-1] if i-1 < len(V2) else "", y[i] - V2[i-1] if i-1 < len(V2) else "",
                     OLP[i-1] if i-1 < len(OLP) else "", Hi[i-1] if i-1 < len(Hi) else "", C1[i-1] if i-1 < len(C1) else "",
                     C2[i-1] if i-1 < len(C2) else ""))

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

    tk.Label(results_window, text="Число итераций:").grid(row=0, column=0)
    tk.Label(results_window, text=number_of_iterations).grid(row=0, column=1)

    tk.Label(results_window, text="Разница между правой границей и последним вычисленным значением:").grid(row=1,column=0)
    tk.Label(results_window, text=difference).grid(row=1, column=1)

    tk.Label(results_window, text="Максимальное значение OLP:").grid(row=2, column=0)
    tk.Label(results_window, text=maxOLP).grid(row=2, column=1)

    tk.Label(results_window, text="Кол-во делений:").grid(row=3, column=0)
    tk.Label(results_window, text=C_1).grid(row=3, column=1)

    tk.Label(results_window, text="Кол-во удвоений:").grid(row=4, column=0)
    tk.Label(results_window, text=C_2).grid(row=4, column=1)

    tk.Label(results_window, text="Макс. значение Hi:").grid(row=5, column=0)
    tk.Label(results_window, text=max_h).grid(row=5, column=1)

    tk.Label(results_window, text="Мин. значение Hi:").grid(row=6, column=0)
    tk.Label(results_window, text=min_h).grid(row=6, column=1)

    tk.Label(results_window, text="Значение x для макс. Hi:").grid(row=9, column=0)
    tk.Label(results_window, text=max_h_x).grid(row=9, column=1)

    tk.Label(results_window, text="Значение x для мин. Hi:").grid(row=10, column=0)
    tk.Label(results_window, text=min_h_x).grid(row=10, column=1)

    plt.cla()

    plt.plot(x, y, label=f'u(x)')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title("График u(x) (численное решение)")
    plt.grid(True)

    plt.draw()

# --------------------------Зависимость от начальной скорости--------------------------
def plot_speed():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())

    u_e_values = [-2, -1, 1, 2.0, y_0]

    plt.figure(figsize=(8, 6), num="Зависимоть решения от начальной скорости")  # Размер фигуры для графика

    for u_e_val in u_e_values:
        func = RungeKutta(h0, x_0, u_e_val, maxCount, epsilonG, a1, a3, m)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x, u = data.T
        plt.plot(x, u, label=f'u = {u_e_val} м/с')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние V0')
    plt.grid(True)
    plt.legend()

    plt.show()  # Показываем график

# --------------------------Отрисовка решения в отдельном окне--------------------------
def plot_u_t():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    step_type = step_type_var.get()
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    m = float(m_entry.get())

    func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3, m)
    data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func.fixedStep(xMax)])
    x, y = data.T

    

    plt.figure(figsize=(8, 6), num="Отрисовка решения в отдельном окне")
    plt.plot(x, y, label=f'u(x)')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.title("График u(x)")
    plt.show()

# --------------------------Зависимость от параметров а1 и а3--------------------------
def plot_param_comparison():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())

    a1_values = [0.2, 0.5, 1.0, a1]
    a3_values = [0.05, 0.1, 0.2, a3]

    
    plt.figure(figsize=(12, 7), num="Зависимость от параметров а1 и а3")

    # Влияние a1 на решение
    plt.subplot(1, 2, 1)
    for a1_val in a1_values:
        func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1_val, a3, m)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x,u = data.T
        plt.plot(x, u, label=f'a1 = {a1_val}')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние a1 на решение')
    plt.grid(True)
    plt.legend()

    # Влияние a3 на решение
    plt.subplot(1, 2, 2)
    for a3_val in a3_values:
        func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3_val, m)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x,u = data.T
        plt.plot(x, u, label=f'a3 = {a3_val}')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние a3 на решение')
    plt.grid(True)
    plt.legend()

    plt.subplots_adjust(hspace=0.5)  # hspace задаёт высоту между строками графиков

    plt.show()

# --------------------------Влияние m на решение--------------------------
def plot_mass():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())

    m_values = [2, 5, 10, m]

    
    plt.figure(figsize=(8, 6), num="Зависимость решение от m")

    # Влияние a1 на решение
    for m_val in m_values:
        func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3, m_val)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x, u = data.T
        plt.plot(x, u, label=f'm = {m_val}')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние m на решение')
    plt.grid(True)
    plt.legend()

    plt.show()

# --------------------------Сравнение вариантов 3 и 4--------------------------
def comparison_3_and_4():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())

    
    plt.figure(figsize=(12, 7), num="Сравнение вариантов 3 и 4")
    plt.subplot(1, 2, 1)
    func_3 = RKVar3(h0, x_0, 5, maxCount, epsilonG, a1, a3, m)
    data = np.array([func_3.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func_3.fixedStep(xMax)])
    x_var_3, y_var_3 = data.T
    plt.plot(x_var_3, y_var_3, linestyle='--', label="Вариант 3")

    func_4 = RungeKutta(h0, x_0, 5, maxCount, epsilonG, a1, a3, m)
    data = np.array([func_4.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func_4.fixedStep(xMax)])
    x_var_4, y_var_4 = data.T
    plt.plot(x_var_4, y_var_4, label="Вариант 4")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title('V > 1 (V = 5)')
    plt.grid(True)
    plt.subplot(1, 2, 2)

    func_3 = RKVar3(h0, x_0, 0.5, maxCount, epsilonG, a1, a3, m)
    data = np.array([func_3.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func_3.fixedStep(xMax)])
    x_var_3, y_var_3 = data.T
    plt.plot(x_var_3, y_var_3, linestyle='--', label="Вариант 3")

    func_4 = RungeKutta(h0, x_0, 0.5, maxCount, epsilonG, a1, a3, m)
    data = np.array([func_4.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func_4.fixedStep(xMax)])
    x_var_4, y_var_4 = data.T
    plt.plot(x_var_4, y_var_4, label="Вариант 4")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title('V < 1 (V = 0.5)')
    plt.grid(True)

    plt.show()

def plot_h0():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())

    y_0 = 0.5

    h0_values = [0.9, 0.5, 0.05, 1e-5]  # Разные значения начального шага

    plt.figure(figsize=(8, 6), num="Зависимость решения от h₀ (начальный шаг)")

    # Влияние h0 на решение
    for h0 in h0_values:
        func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3, m)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x, u = data.T
        plt.plot(x, u, label=f'h₀ = {h0}')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние начального шага (h₀) на решение')
    plt.grid(True)
    plt.legend()

    plt.show()

def plot_Eps():
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    a1 = float(a1_entry.get())
    a3 = float(a3_entry.get())
    step_type = step_type_var.get()
    m = float(m_entry.get())
    epsilonG = float(epsilonG_entry.get())

    y_0 = 0.5

    maxErrorss = [1e-2, 1e-3, 1e-4, 1e-5]  # Разные значения контроля погрешности

    plt.figure(figsize=(8, 6), num="Зависимость решения от ε (контроль погрешности)")

    # Влияние epsilonG на решение
    for maxError in maxErrorss:
        func = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, a1, a3, m)
        data = np.array([func.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func.fixedStep(xMax)])
        x, u = data.T
        plt.plot(x, u, label=f'ε = {maxError}')
    plt.xlabel('x')
    plt.ylabel('Скорость u, м/с')
    plt.title('Влияние контроля погрешности (ε) на решение')
    plt.grid(True)
    plt.legend()

    plt.show()




if __name__ == "__main__":
    root = tk.Tk()
    root.title("Численные методы задание 9 вариант 4")

    frame = ttk.Frame(root)
    frame.pack(padx=10, pady=10)

    root.geometry("1100x800")
    root.minsize(1100, 800)

    window_width = 1100
    window_height = 800

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    maxError_label = ttk.Label(frame, text="Eps. контроля:")
    maxError_label.grid(row=0, column=0)
    maxError_entry = ttk.Entry(frame)
    maxError_entry.grid(row=0, column=1)
    maxError_entry.insert(0, "0.0001")

    maxCount_label = ttk.Label(frame, text="Макс. число итераций:")
    maxCount_label.grid(row=2, column=0)
    maxCount_entry = ttk.Entry(frame)
    maxCount_entry.grid(row=2, column=1)
    maxCount_entry.insert(0, "100000")

    xMax_label = ttk.Label(frame, text="Правая граница:")
    xMax_label.grid(row=3, column=0)
    xMax_entry = ttk.Entry(frame)
    xMax_entry.grid(row=3, column=1)
    xMax_entry.insert(0, "2")

    epsilonG_label = ttk.Label(frame, text="Eps. граничный:")
    epsilonG_label.grid(row=4, column=0)
    epsilonG_entry = ttk.Entry(frame)
    epsilonG_entry.grid(row=4, column=1)
    epsilonG_entry.insert(0, "0.001")

    h0_label = ttk.Label(frame, text="h0:")
    h0_label.grid(row=0, column=2)
    h0_entry = ttk.Entry(frame)
    h0_entry.grid(row=0, column=3)
    h0_entry.insert(0, "0.01")

    u0_label = ttk.Label(frame, text="u0:")
    u0_label.grid(row=1, column=2)
    u0_entry = ttk.Entry(frame)
    u0_entry.grid(row=1, column=3)
    u0_entry.insert(0, "15")

    x0_label = ttk.Label(frame, text="x0:")
    x0_label.grid(row=2, column=2)
    x0_entry = ttk.Entry(frame)
    x0_entry.grid(row=2, column=3)
    x0_entry.insert(0, "0")

    task_var = tk.StringVar()

    step_type_label = ttk.Label(frame, text="Шаг:")
    step_type_label.grid(row=3, column=7)
    step_type_var = tk.StringVar()
    step_type_var.set("Фиксированный")
    step_type_option = ttk.OptionMenu(frame, step_type_var, "Фиксированный","Фиксированный", "Переменный")
    step_type_option.grid(row=3, column=8)

    a1_label = ttk.Label(frame, text="a1:")
    a1_label.grid(row=0, column=7)
    a1_entry = ttk.Entry(frame)
    a1_entry.grid(row=0, column=8)
    a1_entry.insert(0, "1")

    a3_label = ttk.Label(frame, text="a3:")
    a3_label.grid(row=1, column=7)
    a3_entry = ttk.Entry(frame)
    a3_entry.grid(row=1, column=8)
    a3_entry.insert(0, "1")

    m_label = ttk.Label(frame, text="m:")
    m_label.grid(row=2, column=7)
    m_entry = ttk.Entry(frame)
    m_entry.grid(row=2, column=8)
    m_entry.insert(0, "1")


    update_button = ttk.Button(frame, text="Обновить", command=update_plot)
    update_button.grid(row=11, column=1)


    columns = ("i", "xi", "vi", "v2i", "vi-v2i", "|OLP|", "hi", "C1", "C2")
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

    plot_param_button = ttk.Button(frame, text="Влияние a1 и a3", command=plot_param_comparison)
    plot_param_button.grid(row=0, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Влияние m", command=plot_mass)
    plot_param_button.grid(row=1, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Влияние V0", command=plot_speed)
    plot_param_button.grid(row=2, column=30, columnspan=2)

    plot_time_button = ttk.Button(frame, text="Влияние h0", command=plot_h0)
    plot_time_button.grid(row=3, column=30, columnspan=2)

    plot_time_button = ttk.Button(frame, text="Влияние Eps. контроля", command=plot_Eps)
    plot_time_button.grid(row=4, column=30, columnspan=2)

    plot_time_button = ttk.Button(frame, text="Сравнить с вар. 3", command=comparison_3_and_4)
    plot_time_button.grid(row=5, column=30, columnspan=2)

    plot_time_button = ttk.Button(frame, text="График U(x)", command=plot_u_t)
    plot_time_button.grid(row=6, column=30, columnspan=2)

    update_plot()

    root.mainloop()