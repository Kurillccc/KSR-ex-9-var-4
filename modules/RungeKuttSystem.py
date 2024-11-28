import numpy as np

class RungeKuttaSystem:
    def __init__(self, stepSize, initialX, initialU1, initialU2, maxCount, epsilonG, a1, a3, m):
        self.stepSize = stepSize
        self.initialX = initialX
        self.initialU1 = initialU1
        self.initialU2 = initialU2
        self.maxCount = maxCount
        self.epsilonG = epsilonG
        self.a1 = a1
        self.a3 = a3
        self.m = m
        self.V2 = []
        self.OLP = []
        self.Hi = []

        self.C2 = []

    def du1(self, x, u1, u2):
        return u2

    def du2(self, x, u1, u2):
        return -((self.a1/self.m) * u2 + (self.a3/self.m) * (u2**3))

    def calculateNextU(self, x, u1, u2, stepSize):
        k11 = self.du1(x,u1, u2)
        k12 = self.du2(x,u1, u2)
        k21 = self.du1(x + stepSize / 2, u1 + stepSize * k11 / 2, u2 + stepSize * k12 / 2)
        k22 = self.du2(x + stepSize / 2, u1 + stepSize * k11 / 2, u2 + stepSize * k12 / 2)
        k31 = self.du1(x + stepSize / 2, u1 + stepSize * k21 / 2, u2 + stepSize * k22 / 2)
        k32 = self.du2(x + stepSize / 2, u1 + stepSize * k21 / 2, u2 + stepSize * k22 / 2)
        k41 = self.du1(x + stepSize, u1 + stepSize * k31, u2 + stepSize * k32)
        k42 = self.du2(x + stepSize, u1 + stepSize * k31, u2 + stepSize * k32)
        nextU1 = u1 + stepSize * (k11 + 2 * k21 + 2 * k31 + k41)
        nextU2 = u2 + stepSize * (k12 + 2 * k22 + 2 * k32 + k42)
        return (nextU1, nextU2)

    def fixecStep(self, xMax):
        numSteps = int((xMax - self.initialX) / self.stepSize) + 1
        steps = []
        x, u1, u2 = self.initialX, self.initialU1, self.initialU2
        steps.append([x, u1, u2])
        for i in range(1, numSteps):
            x += self.stepSize
            u1, u2 = self.calculateNextU(x - self.stepSize, u1, u2, self.stepSize)
            steps.append([x, u1, u2])
        return steps

    def variableSteps(self, xMax, maxError):
        currentStepSize = self.stepSize
        steps = []
        C1 = 0
        C2 = 0

        currentStep = [self.initialX, self.initialU1, self.initialU2]
        steps.append(currentStep)
        while True:
            nextX = currentStep[0] + currentStepSize
            if nextX > xMax:
                currentStepSize = xMax - currentStep[0]
                continue
            nextU1, nextU2 = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2], currentStepSize)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2], currentStepSize / 2)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0] + currentStepSize / 2, nextU1HalfStep, nextU2HalfStep, currentStepSize / 2)
            errorEstimate1 = abs(nextU1HalfStep - nextU1) / 15
            errorEstimate2 = abs(nextU2HalfStep - nextU2) / 15
            errorEstimate = max(errorEstimate1,errorEstimate2) / 15
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep,nextU2HalfStep])
                self.OLP.append([errorEstimate1 * 16, errorEstimate2 * 16])
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            elif errorEstimate < (maxError / 32):
                C2 += 1
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep, nextU2HalfStep])
                self.OLP.append([errorEstimate1 * 16, errorEstimate2 * 16])
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