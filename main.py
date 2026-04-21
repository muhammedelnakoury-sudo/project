import sys
import math
import numpy as np
from scipy.linalg import lu
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

# ================= SAFE EVAL =================
def f_eval(expr, x):
    expr = expr.replace("^", "**")
    return eval(expr, {"__builtins__": None}, {
        "x": x,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "pi": math.pi, "e": math.e
    })

# ================= ROOT METHODS =================
def bisection(f, xl, xu, tol, it):
    steps = []
    xr_old = 0
    for i in range(1, it+1):
        xr = (xl + xu)/2
        fxr = f_eval(f, xr)
        err = abs((xr-xr_old)/xr)*100 if i > 1 else 100
        steps.append([i, xl, xu, xr, fxr, err])

        if err < tol: break
        if f_eval(f, xl)*fxr < 0: xu = xr
        else: xl = xr
        xr_old = xr
    return steps


def false_position(f, xl, xu, tol, it):
    steps = []
    for i in range(1, it+1):
        xr = xu - (f_eval(f,xu)*(xl-xu))/(f_eval(f,xl)-f_eval(f,xu))
        fxr = f_eval(f, xr)
        steps.append([i, xl, xu, xr, fxr])

        if abs(fxr) < tol: break
        if f_eval(f, xl)*fxr < 0: xu = xr
        else: xl = xr
    return steps


def newton(f, df, x0, tol, it):
    steps = []
    for i in range(1, it+1):
        fx = f_eval(f, x0)
        dfx = f_eval(df, x0)
        if abs(dfx) < 1e-10: break

        x1 = x0 - fx/dfx
        err = abs((x1-x0)/x1)*100
        steps.append([i, x0, fx, dfx, x1, err])

        if err < tol: break
        x0 = x1
    return steps


def secant(f, x0, x1, tol, it):
    steps = []
    for i in range(1, it+1):
        fx0 = f_eval(f,x0)
        fx1 = f_eval(f,x1)
        x2 = x1 - fx1*(x0-x1)/(fx0-fx1)
        err = abs((x2-x1)/x2)*100
        steps.append([i, x0, x1, x2, err])

        if err < tol: break
        x0, x1 = x1, x2
    return steps

# ================= LINEAR =================
def parse_system(text):
    import re
    lines = text.strip().split("\n")
    vars_list = sorted(list(set(c for l in lines for c in l if c.isalpha())))

    A, b = [], []
    for line in lines:
        left, right = line.split("=")
        row = []
        for v in vars_list:
            m = re.findall(rf'([+-]?\d*){v}', left)
            if m:
                coef = m[0]
                if coef in ["", "+"]: coef = 1
                elif coef == "-": coef = -1
                row.append(float(coef))
            else:
                row.append(0)
        A.append(row)
        b.append(float(right))
    return np.array(A), np.array(b), vars_list


def gaussian(A, b):
    steps = []
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i]/A[i][i]
            A[j] -= factor*A[i]
            b[j] -= factor*b[i]
        steps.append((A.copy(), b.copy()))

    x = np.linalg.solve(A, b)
    return steps, x


def gauss_jordan(A, b):
    aug = np.hstack([A, b.reshape(-1,1)])
    n = len(b)

    for i in range(n):
        aug[i] = aug[i]/aug[i,i]
        for j in range(n):
            if i != j:
                aug[j] -= aug[j,i]*aug[i]

    return aug


def cramer(A, b):
    detA = np.linalg.det(A)
    res = []
    for i in range(len(b)):
        Ai = A.copy()
        Ai[:,i] = b
        xi = np.linalg.det(Ai)/detA
        res.append(xi)
    return res


def lu_solve(A, b):
    P,L,U = lu(A)
    y = np.linalg.solve(L,b)
    x = np.linalg.solve(U,y)
    return x

# ================= TABLE =================
def fill_table(table, data, headers):
    table.setRowCount(len(data))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)

    for i,row in enumerate(data):
        for j,val in enumerate(row):
            table.setItem(i,j,QTableWidgetItem(f"{val:.6f}"))

# ================= APP =================
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Methods - Final Project")
        self.resize(1100,650)

        self.setStyleSheet("""
        QWidget { background:white; font-size:14px; }
        QPushButton { background:#e0e0e0; padding:8px; border-radius:6px; }
        QPushButton:hover { background:#d6d6d6; }
        """)

        self.stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        self.menu()
        self.root_page()
        self.linear_page()

    # ===== MENU =====
    def menu(self):
        page = QWidget()
        v = QVBoxLayout(page)

        title = QLabel("Numerical Methods Project")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:26px; font-weight:bold;")

        v.addWidget(title)

        btn1 = QPushButton("Root Finding")
        btn2 = QPushButton("Linear Systems")

        btn1.clicked.connect(lambda:self.stack.setCurrentIndex(1))
        btn2.clicked.connect(lambda:self.stack.setCurrentIndex(2))

        v.addWidget(btn1)
        v.addWidget(btn2)

        self.stack.addWidget(page)

    # ===== ROOT =====
    def root_page(self):
        page = QWidget()
        v = QVBoxLayout(page)

        self.func = QLineEdit("x^3 - x - 2")
        self.dfunc = QLineEdit("3*x^2 - 1")
        self.method = QComboBox()
        self.method.addItems(["Bisection","False Position","Newton","Secant"])

        self.table = QTableWidget()

        btn = QPushButton("Solve")
        btn.clicked.connect(self.solve_root)

        back = QPushButton("Back")
        back.clicked.connect(lambda:self.stack.setCurrentIndex(0))

        v.addWidget(QLabel("f(x):"))
        v.addWidget(self.func)
        v.addWidget(QLabel("f'(x):"))
        v.addWidget(self.dfunc)
        v.addWidget(self.method)
        v.addWidget(btn)
        v.addWidget(self.table)
        v.addWidget(back)

        self.stack.addWidget(page)

    def solve_root(self):
        f = self.func.text()
        df = self.dfunc.text()

        m = self.method.currentText()

        if m=="Bisection":
            data = bisection(f,1,2,0.01,50)
            fill_table(self.table,data,["i","xl","xu","xr","f(xr)","err"])

        elif m=="False Position":
            data = false_position(f,1,2,0.01,50)
            fill_table(self.table,data,["i","xl","xu","xr","f(xr)"])

        elif m=="Newton":
            data = newton(f,df,1,0.01,50)
            fill_table(self.table,data,["i","x","f","df","x_new","err"])

        elif m=="Secant":
            data = secant(f,1,2,0.01,50)
            fill_table(self.table,data,["i","x0","x1","x2","err"])

    # ===== LINEAR =====
    def linear_page(self):
        page = QWidget()
        v = QVBoxLayout(page)

        self.system = QTextEdit("2x + y = 5\nx + y = 3")
        self.method_lin = QComboBox()
        self.method_lin.addItems(["Gaussian","Gauss-Jordan","Cramer","LU"])

        self.table_lin = QTableWidget()

        btn = QPushButton("Solve")
        btn.clicked.connect(self.solve_linear)

        back = QPushButton("Back")
        back.clicked.connect(lambda:self.stack.setCurrentIndex(0))

        v.addWidget(self.system)
        v.addWidget(self.method_lin)
        v.addWidget(btn)
        v.addWidget(self.table_lin)
        v.addWidget(back)

        self.stack.addWidget(page)

    def solve_linear(self):
        A,b,vars_list = parse_system(self.system.toPlainText())
        m = self.method_lin.currentText()

        if m=="Gaussian":
            steps,x = gaussian(A,b)
            result = [[vars_list[i],x[i]] for i in range(len(x))]

        elif m=="Gauss-Jordan":
            aug = gauss_jordan(A,b)
            result = [[vars_list[i],aug[i,-1]] for i in range(len(b))]

        elif m=="Cramer":
            x = cramer(A,b)
            result = [[vars_list[i],x[i]] for i in range(len(x))]

        elif m=="LU":
            x = lu_solve(A,b)
            result = [[vars_list[i],x[i]] for i in range(len(x))]

        fill_table(self.table_lin,result,["Variable","Value"])

# ================= RUN =================
if __name__=="__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())
