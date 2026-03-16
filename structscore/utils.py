# try:
#     from .discourse_utils import *
# except ImportError:  # pragma: no cover - supports running as a script
#     from discourse_utils import *
# from nltk.tokenize import sent_tokenize

# def process_sent_score(str_):
#     str_list = str_.replace("[", "").replace("]", "").replace("\n", "").replace(",", " ").split()
#     return [float(y.strip()) for y in str_list]


# def reweight_score(score_list, row, alpha=1, depth_factor=1):
    
#     summ_segment = literal_eval(row['summ_segments'])
    
#     if len(summ_segment) == 1:
#         summ_segment = summ_segment[0]

#     summ_parsetree = literal_eval(row['summ_tree_parsing'])
    
#     if len(summ_parsetree) == 1:
#         summ_parsetree = summ_parsetree[0][0]
    
#     summ_sent_parse = literal_eval(row['summ_sents'])[0]

#     if summ_parsetree == "NONE":
#         return np.mean(score_list), np.mean(score_list), min(score_list), min(score_list), score_list
#     # parse dicourse tree
#     # recovered edus
#     edus = []
#     for idx, end_id in enumerate(summ_segment):
#         if idx == 0:
#             edus.append(summ_sent_parse[:end_id +1])
#         else:
#             edus.append(summ_sent_parse[summ_segment[idx-1]+1: end_id+1])
#     # print(edus)
#     edu_spans = [recover_list(item) for item in edus]
    
    
#     real_summ = sent_tokenize(row['summary'])
#     if len(real_summ) != len(score_list):
#         return np.mean(score_list), np.mean(score_list), min(score_list), min(score_list), score_list
    
#     assert len(real_summ) == len(score_list)
#     sent2edu_dict = sent2edu_map(real_summ, edu_spans)
#     # print(sent2edu_dict)

#     all_penalty_scores = []
#     all_promote_scores = []
#     all_depth_scores = []

#     all_ono_penalty_scores = []
    
#     for id1 in sent2edu_dict:
#         non_factual_segments = sent2edu_dict[id1]
#         if non_factual_segments == []:
#             all_penalty_scores.append(-1)
#             all_promote_scores.append(-1)
#             all_depth_scores.append(0)
#             all_ono_penalty_scores.append(-1)
#         else:
#             segment_str = str(non_factual_segments[0])+"-"+str(non_factual_segments[-1])
                
#             max_scores = compute_discourse_scores(summ_parsetree, segment_str)

#             # check tree.
#             # print(edu_spans)
#             str_ = translate_preorder(summ_parsetree)
#             tree_root = build_tree(str_)
#             # print_raw_tree(tree_root)
#             found_node, size, depth = find_node_and_compute(tree_root, segment_str)
#             if found_node == None:
#                 start_id, end_id = segment_str.split("-")
#                 depth = int(np.sqrt(int(end_id)-int(start_id)))
#                 tree_root = build_tree(str_)
               
                            
#             all_depth_scores.append(depth)
#             all_penalty_scores.append(max_scores['normalized_max_depth_score'])
#             all_promote_scores.append(max_scores['normalized_max_up_score'])
#             all_ono_penalty_scores.append(max_scores['normalized_max_sent_penalty'])
            
#     # print(all_depth_scores)
    
#     assert len(all_depth_scores) == len(all_penalty_scores)
    
#     mean_penalty = np.mean([y for y in all_penalty_scores if y != -1])
#     mean_promote = np.mean([y for y in all_promote_scores if y != -1])
#     mean_ono_penalty = np.mean([y for y in all_ono_penalty_scores if y != -1])
    
#     # mean_penalty = 0.579
#     # mean_promote = 0.34

    
#     # fix typo.
#     all_penalty_scores = [y if y!= -1 else mean_penalty for y in all_penalty_scores]
#     all_promote_scores = [y if y!= -1 else mean_promote for y in all_promote_scores]
#     all_ono_penalty_scores = [y if y != -1 else mean_ono_penalty for y in all_ono_penalty_scores]
#     #upscale the score.

   
#     # print([1-mean_penalty+all_penalty_scores[i] for i in range(len(all_penalty_scores))]) else score_list[i]*0.9
#     #if score_list[i] <= 0.5 else score_list[i]
    
#     updated_score = [alpha * score_list[i]**(1+mean_penalty-all_penalty_scores[i]) if score_list[i] <= 1 else score_list[i] for i in range(len(score_list))]

#     # updated_score = [alpha * score_list[i]**(1+mean_ono_penalty-all_ono_penalty_scores[i])
#     #                           for i in range(len(score_list))]

#     # 
#     mean_depth_scores = np.mean(all_depth_scores)
  
    
#     #updated_complexity_score = [updated_score[i]**(1+max(0, -all_depth_scores[i]+mean_depth_scores)*depth_factor) for i in range(len(updated_score))]
    
#     updated_complexity_score = [updated_score[i]**(1+all_depth_scores[i]*depth_factor) for i in range(len(updated_score))]
#     return np.mean(updated_complexity_score), np.mean(score_list), min(updated_score), min(score_list), updated_complexity_score
                             
    
# def sent2edu_map(sents, edus):
#     """
#     map the sentences in the summary (split by nltk) to the edus. 
#     """
    
#     sent2edu_dict = {}
#     edus = [y.lower().strip() for y in edus]
#     # initial edu index
#     edu_idx = 0
#     for idx, sent in enumerate(sents):
#         sent = sent.lower().strip()
#         sent2edu_dict[idx] = []
#         cur_edu = edu_idx
#         while cur_edu < len(edus):
#             if edus[cur_edu].strip() in sent.strip() or sent.strip() in edus[cur_edu].strip() or fuzz.partial_ratio(sent.strip(), edus[cur_edu].strip())>= 90:
#                 sent2edu_dict[idx].append(cur_edu+1)
                
#                 cur_edu += 1
#             else:
              
#                 if idx < len(sents)-1 and (edus[cur_edu].strip() in sents[idx-1] + " " + sents[idx] or fuzz.partial_ratio(sents[idx-1] + " " + sents[idx], edus[cur_edu].strip())>= 90):
#                     sent2edu_dict[idx].append(cur_edu+1)
#                     cur_edu += 1
              
#                 edu_idx = cur_edu
#                 break
                
                
#     return sent2edu_dict      
