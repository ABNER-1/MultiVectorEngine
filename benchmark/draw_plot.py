import os
import numpy
import itertools
import matplotlib.pyplot as plt

metrics = {
    "k-nn": {
        "description": "Recall",
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "worst": float("-inf")
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
    # print(filename)
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
                try:
                    id = int(line[0])
                    topk_result[i].append(id)
                except Exception as e:
                    print(e)
                    print(filename, i, j, line)
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
    return numpy.mean(recalls), qps


def remove_useless_point(m_datas):
    result = []
    if len(m_datas) == 0:
        return result
    metrics = sorted(m_datas, key=lambda x: -x[3])
    result.append(metrics[0])
    for metric in metrics:
        previous = result[len(result) - 1]
        if metric[2] > previous[2]:
            result.append(metric)
    return result


def get_metric_data(data_dir, baseline_file, name):
    result_files = []
    files = os.listdir(data_dir)
    for file in files:
        if not os.path.isdir(file):
            result_files.append(os.path.join(data_dir, file))

    metric_datas = []
    for result_file in result_files:
        recall, qps = calc_recall(result_file, baseline_file)
        metric_datas.append(("useless", name, recall, qps))
    metric_datas = remove_useless_point(metric_datas)
    print(name, metric_datas)
    return metric_datas


def draw_ip_recipe(image_name):
    linestyle = create_linestyles([ivf_name, hnsw_name,
                                   nra_ivf_name, nra_hnsw_name,
                                   nra_ivf_omp_name, nra_hnsw_omp_name])
    baseline_file = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/baseline.txt"
    ivf_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/ivf"
    hnsw_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/hnsw"

    nra_ivf_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/nra-ivf"
    nra_hnsw_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/nra-hnsw"
    nra_ivf_omp_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/nra-ivf2"
    nra_hnsw_omp_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-recipe/nra-hnsw2"

    ivf_data = get_metric_data(ivf_dir, baseline_file, ivf_name)
    hnsw_data = get_metric_data(hnsw_dir, baseline_file, hnsw_name)
    nra_ivf_data = get_metric_data(nra_ivf_dir, baseline_file, nra_ivf_name)
    nra_hnsw_data = get_metric_data(nra_hnsw_dir, baseline_file, nra_hnsw_name)
    nra_omp_ivf_data = get_metric_data(nra_ivf_omp_dir, baseline_file, nra_ivf_omp_name)
    nra_omp_hnsw_data = get_metric_data(nra_hnsw_omp_dir, baseline_file, nra_hnsw_name)

    data = {ivf_name: ivf_data, hnsw_name: hnsw_data,
            nra_ivf_name: nra_ivf_data, nra_hnsw_name: nra_hnsw_data,
            nra_ivf_omp_name: nra_omp_ivf_data,
            nra_hnsw_omp_name: nra_omp_hnsw_data}

    create_plot(all_data=data, raw=False, x_log=False, y_log=True,
                xn='k-nn', yn='qps', fn_out=image_name,
                linestyles=linestyle, batch=False)


def draw_ip_glove(image_name):
    linestyle = create_linestyles([ivf_name, hnsw_name,
                                   nra_ivf_name, nra_hnsw_name,
                                   nra_ivf_omp_name, nra_hnsw_omp_name])
    baseline_file = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/baseline.txt"

    ivf_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/ivf"
    hnsw_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/hnsw"
    nra_ivf_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/nra-ivf1"
    nra_hnsw_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/nra-hnsw1"
    nra_ivf_omp_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/nra-ivf"
    nra_hnsw_omp_dir = "/home/abner/workspace/MultiVector/cmake-build-debug/test/ip-glove/nra-hnsw"

    ivf_data = get_metric_data(ivf_dir, baseline_file, ivf_name)
    hnsw_data = get_metric_data(hnsw_dir, baseline_file, hnsw_name)
    nra_ivf_data = get_metric_data(nra_ivf_dir, baseline_file, nra_ivf_name)
    nra_hnsw_data = get_metric_data(nra_hnsw_dir, baseline_file, nra_hnsw_name)
    nra_omp_ivf_data = get_metric_data(nra_ivf_omp_dir, baseline_file, nra_ivf_omp_name)
    nra_omp_hnsw_data = get_metric_data(nra_hnsw_omp_dir, baseline_file, nra_hnsw_name)

    data = {ivf_name: ivf_data, hnsw_name: hnsw_data,
            nra_ivf_name: nra_ivf_data, nra_hnsw_name: nra_hnsw_data,
            nra_ivf_omp_name: nra_omp_ivf_data,
            nra_hnsw_omp_name: nra_omp_hnsw_data}
    create_plot(all_data=data, raw=False, x_log=False, y_log=True,
                xn='k-nn', yn='qps', fn_out=image_name,
                linestyles=linestyle, batch=False)


def draw_l2(image_name):
    linestyle = create_linestyles(["NRA200_16384 IVFFLAT", "NRA200_16384 HNSW",
                                   "NRA800_2048 IVFFLAT", "NRA800_2048 HNSW",
                                   "NRA16384_16384 IVFFLAT", "NRA16384_16384 HNSW",
                                   "NRA2048_2048 IVFFLAT", "NRA2048_2048 HNSW",
                                   "NRA800_16384 IVFFLAT", "NRA800_16384 HNSW",
                                   "NRA16384_16384 IVFFLAT-BATCH", "NRA16384_16384 HNSW-BATCH",
                                   "NRA4096_4096 IVFFLAT", "NRA4096_4096 HNSW",
                                   "NRA2048_2048 IVFFLAT-BATCH", "NRA2048_2048 HNSW-BATCH"])
                                   # "NRA4096_4096 IVFFLAT-BATCH", "NRA4096_4096 HNSW-BATCH"])
    baseline_file = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/baseline.txt"
    ivf_dir1 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/200_16384"
    hnsw_dir1 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/200_16384"
    ivf_dir2 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/800_2048"
    hnsw_dir2 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/800_2048"
    ivf_dir3 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/16384_16384"
    hnsw_dir3 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/16384_16384"
    ivf_dir4 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/2048_2048"
    hnsw_dir4 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/2048_2048"
    ivf_dir5 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/800_16384"
    hnsw_dir5 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/800_16384"
    ivf_dir6 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/16384_16384_batch"
    hnsw_dir6 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/16384_16384_batch"
    ivf_dir8 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/4096_4096"
    hnsw_dir8 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/4096_4096"
    ivf_dir7 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/IVFFLAT/2048_2048_batch"
    hnsw_dir7 = "/home/zilliz/workspace/dev/milvus/MultiVectorEngine/results/l2/HNSW/2048_2048_batch"

    ivf_metric_data1 = get_metric_data(ivf_dir1 , baseline_file, "NRA200_16384 IVFFLAT")
    hnsw_metric_data1 = get_metric_data(hnsw_dir1 , baseline_file, "NRA200_16384 HNSW")
    ivf_metric_data2 = get_metric_data(ivf_dir2 , baseline_file, "NRA800_2048 IVFFLAT")
    hnsw_metric_data2 = get_metric_data(hnsw_dir2 , baseline_file, "NRA800_2048 HNSW")
    ivf_metric_data3 = get_metric_data(ivf_dir3 , baseline_file, "NRA16384_16384 IVFFLAT")
    hnsw_metric_data3 = get_metric_data(hnsw_dir3 , baseline_file, "NRA16384_16384 HNSW")
    ivf_metric_data4 = get_metric_data(ivf_dir4 , baseline_file, "NRA2048_2048 IVFFLAT")
    hnsw_metric_data4 = get_metric_data(hnsw_dir4 , baseline_file, "NRA2048_2048 HNSW")
    ivf_metric_data5 = get_metric_data(ivf_dir5 , baseline_file, "NRA800_16384 IVFFLAT")
    hnsw_metric_data5 = get_metric_data(hnsw_dir5 , baseline_file, "NRA800_16384 HNSW")
    ivf_metric_data6 = get_metric_data(ivf_dir6 , baseline_file, "NRA16384_16384 IVFFLAT-BATCH")
    hnsw_metric_data6 = get_metric_data(hnsw_dir6 , baseline_file, "NRA16384_16384 HNSW-BATCH")
    ivf_metric_data8 = get_metric_data(ivf_dir8 , baseline_file, "NRA4096_4096 IVFFLAT")
    hnsw_metric_data8 = get_metric_data(hnsw_dir8 , baseline_file, "NRA4096_4096 HNSW")
    ivf_metric_data7 = get_metric_data(ivf_dir7 , baseline_file, "NRA2048_2048 IVFFLAT-BATCH")
    hnsw_metric_data7 = get_metric_data(hnsw_dir7 , baseline_file, "NRA2048_2048 HNSW-BATCH")

    # data = {"L2 NRA IVF-Flat": ivf_metric_data, "L2 NRA HNSW": hnsw_metric_data}
    data = {
            # "NRA200_16384 IVFFLAT": ivf_metric_data1, "NRA200_16384 HNSW": hnsw_metric_data1,
            # "NRA800_2048 IVFFLAT": ivf_metric_data2, "NRA800_2048 HNSW": hnsw_metric_data2,
            "NRA16384_16384 IVFFLAT": ivf_metric_data3, "NRA16384_16384 HNSW": hnsw_metric_data3,
            "NRA2048_2048 IVFFLAT": ivf_metric_data4, "NRA2048_2048 HNSW": hnsw_metric_data4,
            "NRA4096_4096 IVFFLAT": ivf_metric_data8, "NRA4096_4096 HNSW": hnsw_metric_data8,
            # "NRA800_16384 IVFFLAT": ivf_metric_data5, "NRA800_16384 HNSW": hnsw_metric_data5,
            "NRA16384_16384 IVFFLAT-BATCH": ivf_metric_data6, "NRA16384_16384 HNSW-BATCH": hnsw_metric_data6,
            "NRA2048_2048 IVFFLAT-BATCH": ivf_metric_data7, "NRA2048_2048 HNSW-BATCH": hnsw_metric_data7}
    create_plot(all_data=data, raw=False, x_log=False, y_log=True,
                xn='k-nn', yn='qps', fn_out=image_name,
                linestyles=linestyle, batch=False)


if __name__ == "__main__":
    ivf_name = "IVF-Flat"
    hnsw_name = "HNSW"
    nra_ivf_name = "NRA IVF-Flat"
    nra_hnsw_name = "NRA HNSW"
    nra_ivf_omp_name = "NRA IVF-Flat with openMP"
    nra_hnsw_omp_name = "NRA HNSW with openMP"

    # draw_ip_recipe("./ip_result_compare.png")
    # draw_ip_glove("./ip_result_glove.png")
    draw_l2("../results/pngs/l2_result_cmp_batch.png")
