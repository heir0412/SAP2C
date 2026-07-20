from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=500, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--flip_test", action='store_true') 
        # 单图推理开关与输入
        self.parser.add_argument('--inference_only', action='store_true',
                         help='Run single-image flow prediction without dataset/GT')
        self.parser.add_argument('--single_image', type=str, default='',
                         help='Path to the RGB image for inference')
        self.parser.add_argument('--single_mask', type=str, default='',
                         help='Path to the binary mask (white=1 region to move)')
        self.parser.add_argument('--single_prompt', type=str, default='',
                         help='Text prompt used when --use_prompts is enabled')
        self.parser.add_argument('--out_size', type=int, default=512,
                         help='Output H=W size for flow & visualization')
# --- single image + single flow generation (new) ---
        self.parser.add_argument('--image_path', type=str, default='', help='path to a single input image')
        self.parser.add_argument('--flow_path', type=str, default='', help='path to a single optical flow file (.pt or .npy)')
        self.parser.add_argument('--semantics_path', type=str, default=None, help='optional semantic map path for video rendering')
        self.parser.add_argument('--n_frames', type=int, default=60, help='number of frames to render for the video/gif')
        self.parser.add_argument('--fps', type=int, default=30, help='fps for mp4/gif export')
        self.parser.add_argument('--save_dir', type=str, default='results/single_video', help='output directory for rendered media')
        self.parser.add_argument('--save_gif', action='store_true', help='also save a GIF alongside MP4')


        self.isTrain = False
