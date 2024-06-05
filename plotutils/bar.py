import matplotlib.pyplot as plt




colors = ['blue', 'orange', 'green']


def barplot(data, bar_width = 0.2):
    groups = data['methods'].to_numpy()
    real_data = data.iloc[:, 1:]
    values = []
    n_groups = len(groups)
    position_groups = []
    for i in range(n_groups):
        row = real_data.iloc[i]
        row = row.dropna().tolist()
        values.append(row)
        num_res = len(row)
        position_groups.append(i + num_res * bar_width / 2)
    
    fig, ax = plt.subplots()
    for i in range(n_groups):
        row = values[i]
        if len(row) == 1:
            ax.bar(position_groups[i], values[0], width=bar_width, color=colors[0], label=groups[0])
        

    