import os
import hashlib
import time
import urllib.request

def download_model(model_path):
    if model_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(os.path.expanduser("~"), ".pgnet_models")
        os.makedirs(cache_dir, exist_ok=True)

        # 使用URL的MD5值作为文件名
        url_md5 = hashlib.md5(model_path.encode()).hexdigest()
        file_ext = os.path.splitext(model_path)[1] or ".onnx"
        cached_file = os.path.join(cache_dir, f"{url_md5}{file_ext}")
        if not os.path.exists(cached_file):
            print(f"正在从{model_path}下载模型...")
            try:
                # 添加下载进度回调函数
                def _progress_hook(count, block_size, total_size):
                    if total_size > 0:
                        percent = min(
                            int(count * block_size * 100 / total_size), 100
                        )
                        downloaded = count * block_size
                        # 计算下载速度（字节/秒）
                        if not hasattr(_progress_hook, "start_time"):
                            _progress_hook.start_time = time.time()
                            _progress_hook.last_size = 0
                            _progress_hook.last_time = _progress_hook.start_time

                        current_time = time.time()
                        interval = current_time - _progress_hook.last_time

                        # 每0.5秒更新一次显示
                        if interval > 0.5 or percent >= 100:
                            size_diff = downloaded - _progress_hook.last_size
                            speed = size_diff / interval if interval > 0 else 0

                            # 转换单位
                            if speed < 1024:
                                speed_str = f"{speed:.2f} B/s"
                            elif speed < 1024 * 1024:
                                speed_str = f"{speed / 1024:.2f} KB/s"
                            else:
                                speed_str = f"{speed / (1024 * 1024):.2f} MB/s"

                            if total_size < 1024 * 1024:
                                size_str = f"{downloaded / 1024:.2f}/{total_size / 1024:.2f} KB"
                            else:
                                size_str = f"{downloaded / (1024 * 1024):.2f}/{total_size / (1024 * 1024):.2f} MB"

                            print(
                                f"\r下载进度: [{percent}%] {size_str} 速度: {speed_str}",
                                end="",
                                flush=True,
                            )

                            _progress_hook.last_size = downloaded
                            _progress_hook.last_time = current_time

                urllib.request.urlretrieve(model_path, cached_file, _progress_hook)
                print("\n模型已下载并保存到{}".format(cached_file))
            except Exception as e:
                raise Exception(f"下载模型时出错: {e}")
        else:
            print(f"使用缓存的模型: {cached_file}")

        return cached_file
    return model_path