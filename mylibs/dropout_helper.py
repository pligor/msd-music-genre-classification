def constructProbs(keep_probs, cur_input_prob, cur_hidden_prob):
    keep_prob_dict = {keep_probs[0]: cur_input_prob}
    
    for i in range(1, len(keep_probs)):
        keep_prob_dict[keep_probs[i]] = cur_hidden_prob
        
    return keep_prob_dict
