from matplotlib import pyplot as plt

def Inch(d_cm):

    d_inch = d_cm / 2.54

    return d_inch

rcCustom_wide = plt.rcParams.copy()
rcCustom_wide["figure.dpi"] = 300
rcCustom_wide["axes.grid"] = True
rcCustom_wide["legend.loc"] = "best"
rcCustom_wide["figure.figsize"] = (Inch(28.58), Inch(12.09))

rcCustom = plt.rcParams.copy()
rcCustom["figure.dpi"] = 150
rcCustom["axes.grid"] = True
rcCustom["grid.alpha"] = 0.2  # ← ADD THIS
rcCustom["grid.linestyle"] = '-'  # ← ADD THIS
rcCustom["legend.loc"] = "best"
rcCustom["figure.figsize"] = (8,6.75)
rcCustom["axes.titlelocation"] = "center"  # alignment of the title: {left, right, center}
rcCustom["axes.titlesize"] = 14   # font size of the axes title
rcCustom["font.size"] = 12