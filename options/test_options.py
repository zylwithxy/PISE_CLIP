from .base_options import BaseOptions
from util import util


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./eval_results/', help='saves results here')
        parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        parser.add_argument('--seg_metric_choice', type= str, default= 'full', choices=['full', 'upper_lower', 'upper'], help= "choose parts of evaluation metric")
        parser.add_argument('--filter_bg', type= util.str2bool, default= False, help="whether to filter background")
        
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1)
        self.isTrain = False

        return parser
