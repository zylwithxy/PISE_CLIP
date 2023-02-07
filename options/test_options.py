from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./eval_results/', help='saves results here')
        parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1)
        self.isTrain = False

        return parser
