from myfuncs import *
import argparse
from datetime import datetime

gc.collect()

parser = argparse.ArgumentParser(description='inputs')

parser.add_argument('--width', type=int, default=1000,
                    help='Width of the canvas')
parser.add_argument('--height', type=int, default=250,
                    help='Height of the canvas')
parser.add_argument('--seqCount', type=int, default=50, help='Sequence count')
parser.add_argument('--timeseries', type=str,
                    default="root.Qloss.targetDevice.test",
                    help='Time series identifier')
parser.add_argument('--raw_template', type=str,
                    default="/root/exp_home/iotdb-cli-0.12.4/tools/dump{}.csv",
                    help='Template for raw data files')
parser.add_argument('--output', type=str, default="/root/output", help='Output directory')

args = parser.parse_args()

output = args.output
timeseries = args.timeseries
width = args.width
height = args.height
seqCount = args.seqCount
raw_template = args.raw_template

# ssim
ssim_res = os.path.join(output, "cache-ssim.csv")
ilts_template = os.path.join(output, "timeQueries", "iotdb", timeseries,
                             "run_0", "iltsResults", "query_{}", "0.csv")
minmax_template = os.path.join(output, "timeQueries", "iotdb", timeseries,
                               "run_0", "minMaxResults", "query_{}", "0.csv")

# query
query_ilts_res = os.path.join(output, "timeQueries", "iotdb", timeseries,
                              "run_0", "iltsResults", "results.csv")
query_minmax_res = os.path.join(output, "timeQueries", "iotdb", timeseries,
                                "run_0", "minMaxResults", "results.csv")

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pngDir=f"tmpCache_{current_time}"
if not os.path.exists(pngDir):
  os.makedirs(pngDir)

################################################################
dpi = 72
lw = 0.7
anti = False
with open(ssim_res, 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['idx', 'ilts', 'minmax'])

  for idx in np.arange(0, seqCount):
    row = [idx]

    raw = raw_template.format(idx)
    ilts = ilts_template.format(idx)
    minmax = minmax_template.format(idx)

    df = pd.read_csv(raw, header=0)
    t = df.iloc[:, 0]
    v = df.iloc[:, 1]
    t = t.to_numpy()
    v = v.to_numpy(dtype='float')
    print(len(t))
    full_frame(width, height, dpi)
    t_min = min(t)  # use raw
    t_max = max(t)  # use raw
    v_min = min(v)  # use raw
    v_max = max(v)  # use raw
    plt.plot(t, v, '-', color='k', linewidth=lw, antialiased=anti)
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    plt.savefig(os.path.join(pngDir,'original-{}-{}.png'.format(idx,timeseries)), backend='agg')
    #         plt.show()
    plt.close()

    df = pd.read_csv(ilts, header=0)
    t = df.iloc[:, 0]
    v = df.iloc[:, 1]
    t = t.to_numpy()
    v = v.to_numpy(dtype='float')
    print(len(t))  # 最后聚合了所以基本都是4w点
    full_frame(width, height, dpi)
    plt.plot(t, v, '-', color='k', linewidth=lw, antialiased=anti)
    plt.xlim(t_min, t_max)  # use raw
    plt.ylim(v_min, v_max)  # use raw
    plt.savefig(os.path.join(pngDir,'iltsCache-{}-{}.png'.format(idx,timeseries)), backend='agg')
    #         plt.show()
    plt.close()
    row.append(
      match(os.path.join(pngDir,'original-{}-{}.png'.format(idx,timeseries)),
            os.path.join(pngDir,'iltsCache-{}-{}.png'.format(idx,timeseries))))

    df = pd.read_csv(minmax, header=0)
    t = df.iloc[:, 0]
    v = df.iloc[:, 1]
    t = t.to_numpy()
    v = v.to_numpy(dtype='float')
    print(len(t))  # 最后聚合了所以基本都是4w点
    full_frame(width, height, dpi)
    plt.plot(t, v, '-', color='k', linewidth=lw, antialiased=anti)
    plt.xlim(t_min, t_max)  # use raw
    plt.ylim(v_min, v_max)  # use raw
    plt.savefig(os.path.join(pngDir,'minMaxCache-{}-{}.png'.format(idx,timeseries)), backend='agg')
    #         plt.show()
    plt.close()
    row.append(
      match(os.path.join(pngDir,'original-{}-{}.png'.format(idx,timeseries)),
            os.path.join(pngDir,'minMaxCache-{}-{}.png'.format(idx,timeseries))))

    print(row)

    writer.writerow(row)

    gc.collect()

print('finish')
gc.collect()

#################################################################
# 画结果

font = 18
lw = 2.5
slw = 1
ms = 9
columnspacing = 0.9
handletextpad = 0.5

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
ax1 = axs[0]
ax2 = axs[1]

# font=18
# lw=2.5
# slw=1
# ms=8
titlepos = -0.43

# Remove space between axes
fig.subplots_adjust(hspace=0.5)  # hspace竖向,wspace横向

##########################################
plt.sca(ax1)

df = pd.read_csv(ssim_res, header=0)
ilts_ssim = df.iloc[:, 1]
minmax_ssim = df.iloc[:, 2]

plt.plot(np.arange(0, df.shape[0]), ilts_ssim, marker='*', markersize=ms,
         linewidth=lw)
plt.plot(np.arange(0, df.shape[0]), minmax_ssim, marker='o', markersize=ms,
         linewidth=lw)

# plt.xlabel("n",fontsize=font)
plt.ylabel("SSIM", fontsize=font)

# ax = plt.gca()
# ax.xaxis.offsetText.set_fontsize(font)
# yScalarFormatter = ScalarFormatter(useMathText=True)
# yScalarFormatter.set_powerlimits((0,0))
# ax.xaxis.set_major_formatter(yScalarFormatter)

# plt.yscale("log")
# plt.xscale("log")

plt.xticks(fontsize=font)
plt.yticks(fontsize=font)

plt.grid()

plt.title('(a) Accuracy', y=titlepos, fontsize=font)

##########################################
plt.sca(ax2)

df = pd.read_csv(query_ilts_res, header=0)
ilts_query = df.iloc[:, 14]  # query time
plt.plot(np.arange(0, df.shape[0]), ilts_query, marker='*', markersize=ms,
         linewidth=lw)

df = pd.read_csv(query_minmax_res, header=0)
minmax_query = df.iloc[:, 14]  # query time
plt.plot(np.arange(0, df.shape[0]), minmax_query, marker='o', markersize=ms,
         linewidth=lw)

# plt.xlabel("m",fontsize=font)
plt.ylabel("time (ms)", fontsize=font)

plt.xticks(fontsize=font)
plt.yticks(fontsize=font)

plt.grid()

plt.title('(b) Query', y=titlepos, fontsize=font)

##########################################
labels = ["ILTSCache", "MinMaxCache"]
fig.legend(ncol=4, fontsize=font, labels=labels, bbox_to_anchor=(0.5, 1.03), \
           loc='upper center', columnspacing=columnspacing, labelspacing=1,
           handletextpad=handletextpad);
plt.savefig('exp-minmaxcache-interaction-new.eps', bbox_inches='tight')
plt.savefig('exp-minmaxcache-interaction-new.png', bbox_inches='tight')
plt.show()
