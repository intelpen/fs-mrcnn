import tensorflow as tf
from ml.mask_rcnn.samples.players_fast.players_fast_dataset import PlayerDataset
from ml.mask_rcnn.mrcnn import visualize
model_filename = "D:\\model\\tf_save_player\\OcclusionFrozzenFast.pb"


with tf.gfile.GFile(model_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and return it
with tf.Graph().as_default() as graph:  # The name var will prefix every op/nodes in your graph  # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=graph)

players_to_predict_DIR = os.path.join("D:\data\ml\mask_rcnn", "players_fast")
dataset = PlayerDataset()
dataset.load_players(players_to_predict_DIR, "val")
dataset.class_names = ['BG', 'person']
dataset.prepare()

# Load and display a random sample
vizualizer = MRCNNVizualizer()
image_id = random.choice(dataset.image_ids)
load_and_display_image(dataset, image_id, vizualizer.ax, display=True)

# Run object detection and benchmark
total_time = 0
for image_id in dataset.image_ids[0:16]:
    total_time += load_and_display_image(dataset, image_id, vizualizer.ax, display=False,
                                         save_filename=f"D:\images_out\{image_id}.png")

print(f"Average {total_time / len(dataset.image_ids)}s/image")

# calculate accuracies

APs, precs, recs = compute_batch_ap(dataset.image_ids[0:16], dataset, config, model)
precs = avg_lst(precs)
recs = avg_lst(recs)
print(precs)
print(recs)
print("mAP @ IoU=50: ", np.mean(APs))
visualize.plot_precision_recall(np.mean(APs), precs, recs)
plt.pause(0.01)
plt.show()
plt.ioff()

r = input()

