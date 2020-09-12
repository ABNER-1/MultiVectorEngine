import os
import numpy
import itertools
import matplotlib.pyplot as plt

metrics = {
    "k-nn": {
        "description": "Recall",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: knn(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf")
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics, 0.1).attrs['mean'],  # noqa
        "worst": float("-inf")
    },
    "rel": {
        "description": "Relative Error",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: rel(true_distances, run_distances, metrics),  # noqa
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: queries_per_second(true_distances, run_attrs),  # noqa
        "worst": float("-inf")
    },
    "distcomps": {
        "description": "Distance computations",
        # "function": lambda true_distances, run_distances,  metrics, run_attrs: dist_computations(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "build": {
        "description": "Build time (s)",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: build_time(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "candidates": {
        "description": "Candidates generated",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: candidates(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "indexsize": {
        "description": "Index size (kB)",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(true_distances, run_attrs),  # noqa
        "worst": float("inf")
    },
    "queriessize": {
        "description": "Index size (kB)/Queries per second (s)",
        # "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(true_distances, run_attrs) / queries_per_second(true_distances, run_attrs), # noqa
        "worst": float("inf")
    }
}


def get_algorithm_name(name, batch_mode):
    if batch_mode:
        return name + "-batch"
    return name


def generate_n_colors(n):
    vs = numpy.linspace(0.4, 1.0, 7)
    colors = [(.9, .4, .4, 1.)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        new_color = max(itertools.product(vs, vs, vs),
                        key=lambda a: min(euclidean(a, b) for b in colors))
        colors.append(new_color + (1.,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(
        zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))
    linestyles = dict((algo, ['--', '-.', '-', ':'][i % 4])
                      for i, algo in enumerate(unique_algorithms))
    markerstyles = dict((algo, ['+', '<', 'o', '*', 'x'][i % 5])
                        for i, algo in enumerate(unique_algorithms))
    faded = dict((algo, (r, g, b, 0.3))
                 for algo, (r, g, b, a) in colors.items())
    return dict((algo, (colors[algo], faded[algo],
                        linestyles[algo], markerstyles[algo]))
                for algo in unique_algorithms)


def get_up_down(metric):
    if metric["worst"] == float("inf"):
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"


def get_plot_label(xm, ym):
    template = ("%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and"
                " to the %(leftright)s is better")
    return template % {"xlabel": xm["description"],
                       "ylabel": ym["description"],
                       "updown": get_up_down(ym),
                       "leftright": get_left_right(xm)}


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als = [], [], []
    # Generate Pareto frontier
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = ((lambda xv, lx: xv > lx)
                  if last_x < 0 else (lambda xv, lx: xv < lx))
    for algo, algo_name, xv, yv in data:
        if not xv or not yv:
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)
        if comparator(xv, last_x):
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
    return xs, ys, ls, axs, ays, als


def create_plot(all_data, raw, x_log, y_log, xn, yn, fn_out, linestyles,
                batch):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        color, faded, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color,
                           ms=7, mew=3, lw=3, linestyle=linestyle,
                           marker=marker)
        handles.append(handle)
        if raw:
            handle2, = plt.plot(axs, ays, '-', label=algo, color=faded,
                                ms=5, mew=2, lw=2, linestyle=linestyle,
                                marker=marker)
        labels.append(get_algorithm_name(algo, batch))

    if x_log:
        plt.gca().set_xscale('log')
    if y_log:
        plt.gca().set_yscale('log')
    plt.gca().set_title(get_plot_label(xm, ym))
    plt.gca().set_ylabel(ym['description'])
    plt.gca().set_xlabel(xm['description'])
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left',
                     bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    if 'lim' in xm:
        plt.xlim(xm['lim'])
    if 'lim' in ym:
        plt.ylim(ym['lim'])
    plt.savefig(fn_out, bbox_inches='tight')
    plt.close()


def read_result_from_file(filename):
    topk_result = []
    with open(filename, 'r') as file:
        line = file.readline().split(' ')
        nq = int(line[0])
        topk = int(line[1])
        time = float(line[2])

        for i in range(nq):
            topk_result.append([])
            for j in range(topk):
                line = file.readline().split(' ')
                id = int(line[0])
                topk_result[i].append(id)
    return nq, time, topk_result


def calc_recall(filename, baseline):
    nq1, time1, topk_results = read_result_from_file(filename)
    nq2, time2, baseline_results = read_result_from_file(baseline)
    qps = nq1 / time1 * 1000
    recalls = []
    for top_result, baseline_result in zip(topk_results, baseline_results):
        topk = top_result
        baseline_topk = baseline_result

        recall_number = 0
        for id in topk:
            if id in baseline_topk:
                recall_number += 1
        recall = recall_number / len(topk)
        recalls.append(recall)
        # result_metric.append(("useless", "algo_name", recall, qps))
    return numpy.mean(recalls), qps


if __name__ == "__main__":
    linestyle = create_linestyles(["ip", "ipnra", "l2"])
    baseline_file = "/home/abner/workspace/MultiVector/cmake-build-debug/test/baseline.txt"
    ip_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip"
    ipnra_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ipnra"
    result_files = []
    files = os.listdir(ip_dir)
    for file in files:
        if not os.path.isdir(file):
            result_files.append(os.path.join(ip_dir, file))
    metrics_datas = []
    for result_file in result_files:
        recall, qps = calc_recall(result_file, baseline_file)
        metrics_datas.append(("useless", "algo_name", recall, qps))
    print("ip", metrics_datas)

    result_files = []
    files = os.listdir(ipnra_dir)
    for file in files:
        if not os.path.isdir(file):
            result_files.append(os.path.join(ipnra_dir, file))
    nra_metrics_datas = []
    for result_file in result_files:
        recall, qps = calc_recall(result_file, baseline_file)
        nra_metrics_datas.append(("useless", "algo_name", recall, qps))
    print("nra", nra_metrics_datas)

    data = {"ip": metrics_datas, "ipnra": nra_metrics_datas}
    create_plot(all_data=data, raw=False, x_log=False, y_log=True,
                xn='k-nn', yn='qps', fn_out="./test1.png",
                linestyles=linestyle, batch=False)
