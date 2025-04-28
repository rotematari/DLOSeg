from der.utils import *
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":

    G = nx.Graph()
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png', cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    mask = mask.astype(np.uint16)
    plt.figure()
    plt.imshow(mask, cmap='gray')
   

    path,G = extract_skeleton(mask)

    pos = nx.get_node_attributes(G, 'pos')
    full_graph = np.array(list(pos.values()))
    plt.figure()
    # plt.imshow(path, cmap='gray')
    nx.draw(G,pos=pos, with_labels=False,node_size=5)
    plt.show()