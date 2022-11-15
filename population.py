from particle import Particle

class Population:
    def __init__(self, pop_size, input_width, input_height, input_channels, max_conv_kernel, max_out_ch, max_pool_kernel, max_fc_neurons, output_dim):
    
        self.particle = []
        for i in range(pop_size):
            self.particle.append(Particle(i, input_width, input_height, input_channels, max_conv_kernel, max_out_ch, max_pool_kernel, max_fc_neurons, output_dim))
                    