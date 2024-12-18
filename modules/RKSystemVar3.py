import numpy as np

# --------------------------Класс для метода РК-4--------------------------
class RKVar3:
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
        return -((self.a1 / self.m) * y + (self.a3 / self.m) * (pow(y, 2)))

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
