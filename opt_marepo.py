import argparse
from pathlib import Path
from distutils.util import strtobool

def _strtobool(x):
    return bool(strtobool(x))
def get_opts():
    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_map', type=str,
                        help='target file for the trained network')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--training_buffer_size', type=int, default=8000000,
                        help='number of patches in the training buffer')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of runs through the training buffer')

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='use homogenous output')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='enforce maximum depth of network predictions')

    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='reprojection error threshold for the visualisation in px')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    # params for the marepo.
    parser.add_argument('--transformer_APR_head', type=_strtobool, default=False,
                        help='use transformer-based APR head')
    parser.add_argument('--head_network_path', type=Path,
                        default=Path(__file__).parent / "logs/wayspots_bears/wayspots_bears.pt",
                        help='file containing pre-trained ACE head weights, (Not used in Marepo training, but for testing)')
    parser.add_argument('--dataset_head_network_path', type=Path,
                        default="",
                        help='path to the pre-trained ACE head weights of entire dataset, e.g. "logs/mapfree/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--preprocessing', type=_strtobool, default=False,
                        help='use pretrained ACE networks to generate scene coordinate maps')
    parser.add_argument('--resume_train', type=_strtobool, default=False,
                        help='True if the model needs to resume training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='highest learning rate of 1 cycle scheduler')
    parser.add_argument("--patience", nargs='+', type=int, default=[200, 50],
                        help='set training schedule for patience [EarlyStopping, reduceLR]')
    parser.add_argument('--transformer_json', type=str, default="../transformer/config/default.json",
                        help='file contain transformer config')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus used for training (pytorch lightning),'
                             'do not use it for now as currently it does not work very well')
    parser.add_argument('--soft_clamping_l1_loss', type=_strtobool, default=False,
                        help='soft clamping on l1 loss when error is high, to reduce clamp losses on unsolvable queries')
    parser.add_argument('--apply_image_encoding', type=_strtobool, default=False,
                        help='apply and fuse image to Transformer')
    parser.add_argument('--apply_ACE_feature_encoding', type=_strtobool, default=False,
                        help='apply and fuse ACE encoder predicted features to Transformer')
    parser.add_argument('--oneCycleScheduler', type=_strtobool, default=False,
                        help='use one cycle scheduler like ACE, instead of reduce on plateau')
    parser.add_argument('--CyclicLRScheduler', type=_strtobool, default=False,
                        help='use cyclic lr scheduler, instead of reduce on plateau')
    parser.add_argument('--CosineAnnealingLR', type=_strtobool, default=False,
                        help='use CosineAnnealing lr scheduler, instead of reduce on plateau')
    parser.add_argument('--CosineAnnealingWarmRestarts', type=_strtobool, default=False,
                        help='use CosineAnnealingWarmRestarts lr scheduler, instead of reduce on plateau')
    parser.add_argument('--learning_rate_max', type=float, default=0.002,
                        help='highest learning rate of 1 cycle scheduler')
    parser.add_argument('--cosine_similarity_fuse', type=_strtobool, default=False,
                        help='use cosine similarity fused confidence map instead of dot product')
    parser.add_argument('--resume_from_pretrain', type=_strtobool, default=False,
                        help='resume training from a pre-trained model')
    parser.add_argument('--pretrain_model_path', type=str,
                        help='path to pretrained model')

    parser.add_argument('--refine_DSAC', type=_strtobool, default=False,
                        help='Refine DSAC pose using marepo')

    parser.add_argument('--check_val_every_n_epoch', type=int, default=5,
                        help='run validations set every n epoch')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                        help='sanity check on validation before training, 0: no check, -1: check all val set')
    parser.add_argument('--random_rescale_sc', type=_strtobool, default=False,
                        help='')

    # dataset loader related args
    parser.add_argument('--dataset_path', type=Path,
                        default="",
                        help='path to the dataset folder, e.g. "~/storage/map_free_training_scenes/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--finetune', type=_strtobool, default=False,
                        help='finetune the mapping set of a specific test scenes')
    parser.add_argument('--train_mapping_query', type=_strtobool, default=False,
                        help='train marepo with mapping and query data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of train set batch size')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='number of val set batch size')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='number of test set batch size')
    parser.add_argument('--load_scheme2_sc_map', type=_strtobool, default=True,
                        help='use saved SC maps (subtract mean) and GT pose (subtract mean)'
                             'instead of use original SC map and GT pose')
    parser.add_argument('--load_scheme3_sc_map', type=_strtobool, default=True,
                        help='use additional saved augmented SC maps (subtract mean) and GT pose (subtract mean)')
    parser.add_argument('--marepo_sc_augment', type=_strtobool, default=False,
                        help='apply additional data augmentation for Marepo, such as to preprocessed SC maps ')
    parser.add_argument('--non_mapfree_dataset_naming', type=_strtobool, default=False,
                        help='A temp fix for data augmentation that is not mapfree data, to be unified later')
    parser.add_argument('--jitter_trans', type=float, default=1.0,
                        help='translation jitter range (in meters)')
    parser.add_argument('--jitter_rot', type=float, default=15.0,
                        help='ratation jitter range (in degrees)')
    parser.add_argument('--fuse_mapping_confidence', type=_strtobool, default=False,
                        help='fuse mapping confidence (pitch 3)')
    parser.add_argument('--random_mapping_buffers', type=_strtobool, default=False,
                        help='select randomly among pre-stored 10 mappping buffers for train set')
    parser.add_argument('--all_mapping_buffers', type=_strtobool, default=False,
                        help='select all 10 mappping buffers for train set')
    parser.add_argument('--center_crop', type=_strtobool, default=False,
                        help='Flag for datasetloader indicating images need center crop to make them proportional in size to MapFree data')
    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')
    parser.add_argument('--use_val_half', type=_strtobool, default=True,
                        help='val with half precision')
    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')
    parser.add_argument('--trainskip', type=int, default=1,
                        help='uniformly subsample train set by 1/trainskip')
    parser.add_argument('--testskip', type=int, default=1,
                        help='uniformly subsample val/test set by 1/testskip')
    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')
    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')
    parser.add_argument('--load_rgb', type=_strtobool, default=False,
                        help='Use 3 rgb channel images instead of using 1 channel gray image.')

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='split the training sequence in this number of clusters. disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='train on images part of this cluster. required only if --num_clusters is set.')

    # simple integration testing for code cleaning
    parser.add_argument('--integration_test', type=_strtobool, default=False,
                        help='simple integration test for code cleaning')


    return parser.parse_args()
