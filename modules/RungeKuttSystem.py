import numpy as np

class RungeKutta:
    def __init__(self, stepSize, initialX, initialY, maxCount, epsilonG, a1, a3, m):
        self.epsilonG = epsilonG
        self.maxCount = maxCount
        self.stepSize = stepSize
        self.initialX = initialX
        self.initialY = initialY
        self.V2 = []
        self.OLP = []
        self.Hi = []
        self.a1 = a1
        self.a3 = a3
        self.m = m
        self.C1 = []
        self.C2 = []

    def function(self, x, y):
        return -((self.a1 / self.m) * x + (self.a3 / self.m) * (x ** 3))

    def calculateNextY(self, x, y, stepSize):
        k1 = self.function(x, y)
        k2 = self.function(x + stepSize / 2, y + k1 * stepSize / 2)
        k3 = self.function(x + stepSize / 2, y + k2 * stepSize / 2)
        k4 = self.function(x + stepSize, y + k3 * stepSize)
        nextY = y + stepSize * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return nextY

    def fixedStep(self, xMax):
        numSteps = int((xMax - self.initialX) / self.stepSize) + 1
        steps = []
        x, y = self.initialX, self.initialY
        steps.append([x, y])
        for i in range(1, numSteps):
            x = x + self.stepSize
            y = self.calculateNextY(x - self.stepSize, y, self.stepSize)
            steps.append([x, y])

        return steps

    def variableStep(self, xMax, maxError):
        currentStepSize = self.stepSize
        steps = []
        C1 = 0
        C2 = 0

        currentStep = [self.initialX, self.initialY]
        steps.append(currentStep)
        while True:
            nextX = currentStep[0] + currentStepSize
            if nextX > xMax:
                currentStepSize = xMax - currentStep[0]
                continue
            nextY = self.calculateNextY(currentStep[0], currentStep[1], currentStepSize)
            nextYHalfStep = self.calculateNextY(currentStep[0], currentStep[1], currentStepSize / 2)
            nextYHalfStep = self.calculateNextY(currentStep[0] + currentStepSize / 2, nextYHalfStep, currentStepSize / 2)
            errorEstimate = abs((nextYHalfStep - nextY) / 15)
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextY]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append(nextYHalfStep)
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            elif errorEstimate < (maxError / 32):
                C2 += 1
                nextStep = [nextX, nextY]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append(nextYHalfStep)
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)
                currentStepSize *= 2

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break
            else:
                currentStepSize /= 2
                C1 += 1
            if len(steps) != self.maxCount:
                continue
            break

        return steps

def rk4_adaptive(u0, t0, t_max, h0, tol, f):
    """
    Адаптивный метод Рунге-Кутты 4-го порядка.

    Параметры:
        u0   - начальное значение функции (скаляр или вектор),
        t0   - начальный момент времени,
        t_max - конечный момент времени,
        h0   - начальный шаг интегрирования,
        tol  - требуемая точность,
        f    - функция правой части уравнения f(u, t).

    Возвращает:
        t_vals - массив значений времени,
        u_vals - массив значений функции u.
    """
    t = t0
    u = u0
    h = h0

    t_vals = [t]
    u_vals = [u]

    while t < t_max:
        if t + h > t_max:  # Уменьшаем шаг, чтобы не выйти за предел
            h = t_max - t

        # Шаг Рунге-Кутты 4-го порядка
        k1 = h * f(u)
        k2 = h * f(u + k1 / 2)
        k3 = h * f(u + k2 / 2)
        k4 = h * f(u + k3)

        u_full_step = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Половинный шаг для оценки ошибки
        h_half = h / 2
        k1_half = h_half * f(u)
        k2_half = h_half * f(u + k1_half / 2)
        k3_half = h_half * f(u + k2_half / 2)
        k4_half = h_half * f(u + k3_half)

        u_half_step_1 = u + (k1_half + 2 * k2_half + 2 * k3_half + k4_half) / 6

        k1_half = h_half * f(u_half_step_1)
        k2_half = h_half * f(u_half_step_1 + k1_half / 2)
        k3_half = h_half * f(u_half_step_1 + k2_half / 2)
        k4_half = h_half * f(u_half_step_1 + k3_half)

        u_half_step_2 = u_half_step_1 + (k1_half + 2 * k2_half + 2 * k3_half + k4_half) / 6

        # Оценка локальной ошибки
        error = np.abs(u_half_step_2 - u_full_step) / 15

        if error < tol:
            # Прием шага: точность удовлетворяет условию
            t += h
            u = u_full_step
            t_vals.append(t)
            u_vals.append(u)

        # Коррекция шага
        if error != 0:
            h = h * min(2, max(0.1, 0.9 * (tol / error) ** 0.25))
        else:
            h *= 2  # Если ошибка нулевая, увеличиваем шаг в 2 раза

    return np.array(t_vals), np.array(u_vals)
