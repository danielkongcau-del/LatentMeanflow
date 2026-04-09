import time
import torch
import numpy as np
from choices import choose_net
def run_one_round(
    model,
    dummy_input,
    device="cuda",
    warmup_iters=5,
    test_iters=50,
    include_postprocess=False,
    use_amp=False,
):
    # 预热
    for _ in range(warmup_iters):
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(dummy_input)
            else:
                out = model(dummy_input)

            if isinstance(out, dict):
                out = out["out"]

            if include_postprocess:
                _ = torch.argmax(out, dim=1)

        if device == "cuda":
            torch.cuda.synchronize()

    # 正式测
    times = []
    for _ in range(test_iters):
        start = time.time()
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(dummy_input)
            else:
                out = model(dummy_input)

            if isinstance(out, dict):
                out = out["out"]

            if include_postprocess:
                _ = torch.argmax(out, dim=1)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    times = np.array(times)
    avg_latency = times.mean()
    p95_latency = np.percentile(times, 95)
    total_time = times.sum()

    return avg_latency, p95_latency, total_time


def benchmark_segmentation_multi_round(
    model,
    img_size=(512, 512),
    device="cuda",
    batch_size=1,
    num_rounds=10,      # 你要的10轮
    warmup_iters=5,
    test_iters=50,
    include_postprocess=False,
    use_amp=False,
):
    model = model.to(device)
    model.eval()

    h, w = img_size
    dummy_input = torch.randn(batch_size, 3, h, w, device=device)

    round_avg_latencies = []
    round_p95_latencies = []
    round_throughputs = []

    for r in range(num_rounds):
        avg_lat, p95_lat, total_time = run_one_round(
            model,
            dummy_input,
            device=device,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
            include_postprocess=include_postprocess,
            use_amp=use_amp,
        )

        # 本轮的吞吐量：这一轮总共测了 test_iters * batch_size 张图
        total_imgs = test_iters * batch_size
        throughput = total_imgs / total_time

        round_avg_latencies.append(avg_lat)
        round_p95_latencies.append(p95_lat)
        round_throughputs.append(throughput)

        print(f"Round {r+1}/{num_rounds}: "
              f"avg={avg_lat*1000:.2f} ms, "
              f"p95={p95_lat*1000:.2f} ms, "
              f"throughput={throughput:.2f} img/s")

    # 最终的“更公平的平均”
    final_avg_latency = float(np.mean(round_avg_latencies))
    final_p95_latency = float(np.mean(round_p95_latencies))
    final_throughput = float(np.mean(round_throughputs))

    print("\n===== Final (averaged over rounds) =====")
    print(f"Image size: {h}x{w}, batch_size={batch_size}")
    print(f"Avg latency (mean of rounds): {final_avg_latency*1000:.2f} ms")
    print(f"P95 latency (mean of rounds): {final_p95_latency*1000:.2f} ms")
    print(f"Throughput (mean of rounds): {final_throughput:.2f} img/s")

    return {
        "avg_latency_s": final_avg_latency,
        "p95_latency_s": final_p95_latency,
        "throughput_images_per_s": final_throughput,
        "round_avg_latencies_s": round_avg_latencies,
        "round_p95_latencies_s": round_p95_latencies,
        "round_throughputs": round_throughputs,
    }



if __name__ == "__main__":
    name = "TransUNetHA"  # 改成 "unet"、"deeplabv3" 之类的真实名字  SegFormerHA
    # 4.SegFormer    SegFormerHA
    # 5.TransUNet    TransUNetHA
    # 6.SwinUNet   SwinUNetHA
    # 12.CMTFNet    CMTFNetHA
    # 11.BRAUNet   BRAUNetHA
    # 13.ScaleFormer  ScaleFormerHA


    if name != "":  # 只有有名字才去构建
        model = choose_net(name=name, out_channels=2)
    else:
        raise ValueError("请先给 name 一个正确的模型名称")

    benchmark_segmentation_multi_round(
        model,
        img_size=(512, 512),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
        num_rounds=10,
        warmup_iters=5,
        test_iters=10,
        include_postprocess=True,
    )

