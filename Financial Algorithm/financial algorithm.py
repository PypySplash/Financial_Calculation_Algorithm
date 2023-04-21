import tkinter as tk
from lookback_put import lookback_put
from lookback_put import MonteCarlo_lookback_puts
from CRR import CRR_binomial_tree_model
# from BS_MonteCarlo import Black_Scholes
from BS_MonteCarlo import MonteCarlo
from avgcall_finalgorithm import arithmetic_avg_call
from avgcall_finalgorithm import MonteCarlo_arithmetic_avg_call


root = tk.Tk()
root.title('Financial Algorithm')
canvas1 = tk.Canvas(root, width=500, height=500, relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='Calculate the Option value')
label1.config(font=('helvetica', 14))
canvas1.create_window(100, 25, window=label1)

labelSt = tk.Label(root, text='Type your St:')
labelSt.config(font=('helvetica', 10))
canvas1.create_window(10, 50, window=labelSt)
entrySt = tk.Entry(root)
canvas1.create_window(150, 50, window=entrySt)

labelK = tk.Label(root, text='Type your K:')
labelK.config(font=('helvetica', 10))
canvas1.create_window(10, 70, window=labelK)
entryK = tk.Entry(root)
canvas1.create_window(150, 70, window=entryK)

labelr = tk.Label(root, text='Type your r:')
labelr.config(font=('helvetica', 10))
canvas1.create_window(10, 90, window=labelr)
entryr = tk.Entry(root)
canvas1.create_window(150, 90, window=entryr)

labelq = tk.Label(root, text='Type your q:')
labelq.config(font=('helvetica', 10))
canvas1.create_window(10, 110, window=labelq)
entryq = tk.Entry(root)
canvas1.create_window(150, 110, window=entryq)

label_sigma = tk.Label(root, text='Type your sigma:')
label_sigma.config(font=('helvetica', 10))
canvas1.create_window(10, 130, window=label_sigma)
entry_sigma = tk.Entry(root)
canvas1.create_window(150, 130, window=entry_sigma)

labelt = tk.Label(root, text='Type your t:')
labelt.config(font=('helvetica', 10))
canvas1.create_window(10, 150, window=labelt)
entryt = tk.Entry(root)
canvas1.create_window(150, 150, window=entryt)

labelT = tk.Label(root, text='Type your T:')
labelT.config(font=('helvetica', 10))
canvas1.create_window(10, 170, window=labelT)
entryT = tk.Entry(root)
canvas1.create_window(150, 170, window=entryT)

labeln = tk.Label(root, text='Type your n:')
labeln.config(font=('helvetica', 10))
canvas1.create_window(10, 190, window=labeln)
entryn = tk.Entry(root)
canvas1.create_window(150, 190, window=entryn)

label_Smaxt = tk.Label(root, text='Type your Smaxt:')
label_Smaxt.config(font=('helvetica', 10))
canvas1.create_window(10, 210, window=label_Smaxt)
entry_Smaxt = tk.Entry(root)
canvas1.create_window(150, 210, window=entry_Smaxt)

label_callorput = tk.Label(root, text='Type your callorput:')
label_callorput.config(font=('helvetica', 10))
canvas1.create_window(10, 230, window=label_callorput)
entry_callorput = tk.Entry(root)
canvas1.create_window(150, 230, window=entry_callorput)

label_AorE = tk.Label(root, text='Type your AorE:')
label_AorE.config(font=('helvetica', 10))
canvas1.create_window(10, 250, window=label_AorE)
entry_AorE = tk.Entry(root)
canvas1.create_window(150, 250, window=entry_AorE)

labelns = tk.Label(root, text='Type your ns:')
labelns.config(font=('helvetica', 10))
canvas1.create_window(10, 270, window=labelns)
entryns = tk.Entry(root)
canvas1.create_window(150, 270, window=entryns)

labelnr = tk.Label(root, text='Type your nr:')
labelnr.config(font=('helvetica', 10))
canvas1.create_window(10, 290, window=labelnr)
entrynr = tk.Entry(root)
canvas1.create_window(150, 290, window=entrynr)


def getOptionValue():
    St = entrySt.get()
    K = entryK.get()
    r = entryr.get()
    q = entryq.get()
    sigma = entry_sigma.get()
    t = entryt.get()
    T = entryT.get()
    n = entryn.get()
    Smaxt = entry_Smaxt.get()
    callorput = entry_callorput.get()
    AorE = entry_AorE.get()
    ns = entryns.get()
    nr = entrynr.get()

    CRR = CRR_binomial_tree_model(float(St), float(K), float(r), float(q), float(sigma), float(T), int(n), callorput, AorE)
    Monte_CRR = MonteCarlo(float(St), float(K), float(r), float(q), float(sigma), float(T), callorput, int(ns), int(nr))
    lookback = lookback_put(float(St), float(r), float(q), float(sigma), float(t), float(T), int(n), float(Smaxt), AorE)
    Monte_lookback = MonteCarlo_lookback_puts(float(St), float(r), float(q), float(sigma), float(t), float(T), float(Smaxt), int(n), int(ns), int(nr))
    avg_call = arithmetic_avg_call(float(St), float(K), float(r), float(q), float(sigma), float(t), float(T), int(n), float(Smaxt), AorE)
    MonteCarlo_avg_call = MonteCarlo_arithmetic_avg_call(float(St), float(K), float(r), float(q), float(sigma), float(t), float(T), int(n), float(Smaxt), int(ns), int(nr))
    
    label_lookback_input = tk.Label(root, text='The Option Value of' + ' lookback & MonteCarlo ' + 'is:', font=('helvetica', 10))
    canvas1.create_window(150, 330, window=label_lookback_input)
    label_lookback_output = tk.Label(root, text=lookback, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 350, window=label_lookback_output)
    label_Monte_lookback_output = tk.Label(root, text=Monte_lookback, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 370, window=label_Monte_lookback_output)  # 這個會跑比較久，先註解掉

    label_CRR_input = tk.Label(root, text='The Option Value of'+' CRR & MonteCarlo '+'is:', font=('helvetica', 10))
    canvas1.create_window(150, 390, window=label_CRR_input)
    label_CRR_output = tk.Label(root, text=CRR, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 410, window=label_CRR_output)
    label_MonteCarlo = tk.Label(root, text=Monte_CRR, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 430, window=label_MonteCarlo)

    label_avg_input = tk.Label(root, text='The Option Value of'+' avg & MonteCarlo '+'is:', font=('helvetica', 10))
    canvas1.create_window(150, 450, window=label_avg_input)
    label_avg_output = tk.Label(root, text=avg_call, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 470, window=label_avg_output)
    label_MonteCarlo_avg_output = tk.Label(root, text=MonteCarlo_avg_call, font=('helvetica', 10, 'bold'))
    canvas1.create_window(150, 490, window=label_MonteCarlo_avg_output)  # 這個會跑比較久，先註解掉


button1 = tk.Button(text='Get the Option Value', command=getOptionValue, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas1.create_window(150, 310, window=button1)

root.mainloop()
