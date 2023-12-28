import os


def main():
    bvh_dir = "E:/Datasety/kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH"

    if not os.path.isdir(bvh_dir):
        print(f"Directory {bvh_dir} does not exist.")
        return

    for folder in os.listdir(bvh_dir):
        folder_path = os.path.join(bvh_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.bvh'):
                file_path = os.path.join(folder_path, filename)
                os.system(f"bvh-converter {file_path}")
                print(f"Converted: {file_path}")


if __name__ == "__main__":
    main()
