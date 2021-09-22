import numpy as np
import graphics
import rover
import sys 

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # Initialization for forward message
    forward_messages[0] = rover.Distribution({})
    # populate the Distribution dict for all possible hidden states 
    for z0 in all_possible_hidden_states:
        prior_prob = prior_distribution[z0]
        if observations[0] != None:
            observation_prob = observation_model(z0)[observations[0]]
        else:
            observation_prob = 1
        if prior_prob * observation_prob != 0:
            forward_messages[0][z0] = prior_prob * observation_prob
    forward_messages[0].renormalize()

    # use recursive relationship to compute all forward messages 
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            if observations[i] != None:
                observation_prob = observation_model(zi)[observations[i]]
            else:
                observation_prob = 1 
            sum_of_prev_messages = 0
            for prev_z in forward_messages[i-1].keys():
                sum_of_prev_messages += forward_messages[i-1][prev_z] * transition_model(prev_z)[zi]
            if observation_prob * sum_of_prev_messages != 0:
                forward_messages[i][zi] = observation_prob * sum_of_prev_messages
        forward_messages[i].renormalize()
    
    # Initialization for backward messages 
    backward_messages[num_time_steps-1] = rover.Distribution({})
    for last_z in all_possible_hidden_states:
        backward_messages[num_time_steps-1][last_z] = 1
    
    # use recursive relationship to compute all backward messages
    for i in range(num_time_steps-2, -1, -1):
        backward_messages[i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            sum_of_prev_messages = 0
            for subsequent_z in backward_messages[i+1].keys():
                if observations[i+1] != None:
                    observation_prob = observation_model(subsequent_z)[observations[i+1]]
                else:
                    observation_prob = 1 
                sum_of_prev_messages += backward_messages[i+1][subsequent_z] * observation_prob * transition_model(zi)[subsequent_z]
            if sum_of_prev_messages != 0:
                backward_messages[i][zi] = sum_of_prev_messages
        backward_messages[i].renormalize()        

    for i in range(num_time_steps):
        marginals[i] = rover.Distribution({})
        total_of_marginals = 0
        for zi in all_possible_hidden_states:
            total_of_marginals += forward_messages[i][zi] * backward_messages[i][zi]
        for zi in all_possible_hidden_states:
            if forward_messages[i][zi] * backward_messages[i][zi] != 0:
                marginals[i][zi] = (forward_messages[i][zi] * backward_messages[i][zi]) / total_of_marginals

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # initialization of the lists
    num_time_steps = len(observations)
    W = [None] * num_time_steps 
    estimated_hidden_states = [None] * num_time_steps
    path_trellis = [None] * num_time_steps
    # initialization of the messages 
    W[0] = rover.Distribution({})
    for z0 in all_possible_hidden_states:
        prior_prob = prior_distribution[z0]
        if observations[0] != None:
            observation_prob = observation_model(z0)[observations[0]]
        else:
            observation_prob = 1
        if prior_prob !=0 and observation_prob !=0:
            W[0][z0] = np.log(prior_prob) + np.log(observation_prob)
    # recursion steps 
    for i in range(1, num_time_steps):
        W[i] = rover.Distribution({})
        path_trellis[i] = {}
        for zi in all_possible_hidden_states:
            if observations[i] != None:
                observation_prob = observation_model(zi)[observations[i]]
            else:
                observation_prob = 1 
            max_prev_trans_prob = float('-inf')
            for prev_z in W[i-1].keys():
                if transition_model(prev_z)[zi] != 0:
                    prev_trans_prob = np.log(transition_model(prev_z)[zi]) + W[i-1][prev_z]
                    if prev_trans_prob > max_prev_trans_prob:
                        max_prev_trans_prob = prev_trans_prob
                        # remember the transition from which previous state is the most likely 
                        path_trellis[i][zi] = prev_z
            if observation_prob != 0:
                W[i][zi] = np.log(observation_prob) + max_prev_trans_prob
    
    # obtain final prediction sequence
    # obtain the final hidden state prediction
    estimated_hidden_states[num_time_steps-1] = max(W[num_time_steps-1], key=W[num_time_steps-1].get)
    # back track using the path trellis to find the rest of the predictions
    for i in range(num_time_steps-2, -1, -1):
        estimated_hidden_states[i] = path_trellis[i+1][estimated_hidden_states[i+1]]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True 
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
        print('Missing activated')
    else:
        filename = 'test.txt'
        print('Missing not activated')
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    if missing_observations == False:
        print("Most likely parts of marginal at time %d:" % (timestep))
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print("Most likely parts of marginal at time %d:" % (30))
        print(sorted(marginals[30].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # compute error probabilities 
    correct = 0
    for i in range(len(estimated_states)):
        if estimated_states[i] == hidden_states[i]:
            correct += 1 
    print("Viterbi Error Probability = %.4f" %(1-correct/100)) 

    correct = 0 
    for i in range(len(marginals)):
        z_pred = max(marginals[i], key=marginals[i].get)
        if z_pred == hidden_states[i]:
            correct += 1 
    print("Forward-Backward Error Probability = %.4f" %(1-correct/100)) 

    # search for possible violations, start off with a violation involving 'stay' action 
    for i in range(1, len(marginals)):
        z_pred = max(marginals[i], key=marginals[i].get)
        prev_z_pred = max(marginals[i-1], key=marginals[i-1].get)
        if z_pred[2] == 'stay' and (z_pred[0] != prev_z_pred[0] or z_pred[1] != prev_z_pred[1]):
            print('Violation', i, prev_z_pred, z_pred)

    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
