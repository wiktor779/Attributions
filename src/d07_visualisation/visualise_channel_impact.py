import matplotlib.pyplot as plt


def visualize_channel_impact(impact_dict, filename):
    plot = plt.bar(impact_dict.keys(), impact_dict.values())

    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width() / 2.,
                 1.002 * height, height, ha='center', va='bottom')

    plt.title(filename)
    plt.xlabel("KANAŁ")
    plt.ylabel("WPŁYW (%)")

    filepath = f'../../results/{filename}.png'
    plt.savefig(filepath)
    print(f'Saved results to: {filepath}')
    plt.show()
