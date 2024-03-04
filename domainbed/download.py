import os
import gdown
import tarfile
from zipfile import ZipFile
import argparse


def stage_path(data_dir, name):
    """
    在指定的目录下创建一个子目录，
    并返回该子目录的完整路径。
    如果该子目录已经存在，
    则不会进行任何操作，直
    接返回已存在的路径。
    """
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    """
    从给定的 url 下载文件，并将其解压到指定的目标目录 dst 中。
    """
    # 下载指定的 url 文件到目标路径 dst 中。quiet=False 表示在下载过程中输出详细信息。
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))
    os.rmdir(full_path)
    os.rename(os.path.join(data_dir, "kfold"),
              full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='./testData', required=False)
    args = parser.parse_args()

    download_pacs(args.data_dir)
