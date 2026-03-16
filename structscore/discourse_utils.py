"""Utilities for discourse-tree parsing and StructScore aggregation."""

from ast import literal_eval
from collections import defaultdict, deque

import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize


def align_sentences_and_segments_fuzzy(sentences, segments, threshold=80):
    """Align summary sentences to EDU segments with fuzzy string overlap."""
    alignment = defaultdict(list)

    for i, sentence_a in enumerate(sentences):
        for j, segment_b in enumerate(segments):
            similarity_score = fuzz.partial_ratio(segment_b, sentence_a)

            if similarity_score >= threshold:
                alignment[i].append(j + 1)

    return dict(alignment)


def remove_overlapping_values(alignment):
    """Keep the earliest assignment for each aligned EDU index."""
    used_indices = set()
    cleaned_alignment = defaultdict(list)

    for key in sorted(alignment.keys()):
        for value in alignment[key]:
            if value not in used_indices:
                cleaned_alignment[key].append(value)
                used_indices.add(value)

    return dict(cleaned_alignment)


def fix_alignment(alignment):
    """Repair noisy alignments by shifting outliers and filling short gaps."""
    corrected_alignment = {key: alignment[key][:] for key in alignment}
    keys = sorted(corrected_alignment.keys())
    threshold = 1

    for idx, key in enumerate(keys):
        values = sorted(corrected_alignment[key])
        if not values:
            continue

        values_to_move = []
        for i in range(1, len(values)):
            if values[i] - values[i - 1] > threshold:
                values_to_move.extend(values[i:])
                break

        if values_to_move and idx + 1 < len(keys):
            next_key = keys[idx + 1]
            corrected_alignment[next_key].extend(values_to_move)
            corrected_alignment[key] = values[:values.index(values_to_move[0])]

    for key in corrected_alignment:
        values = sorted(set(corrected_alignment[key]))
        if not values:
            continue
        min_value, max_value = min(values), max(values)
        expected_sequence = set(range(min_value, max_value + 1))
        missing_values = expected_sequence - set(values)
        if missing_values:
            corrected_alignment[key].extend(missing_values)
            corrected_alignment[key].sort()

    return corrected_alignment


def tree_to_graph(root):
    """Convert a binary discourse tree into an undirected graph."""
    graph = nx.Graph()

    def helper(node):
        if node:
            graph.add_node(node.value)
            if node.left:
                graph.add_edge(node.value, node.left.value)
                helper(node.left)
            if node.right:
                graph.add_edge(node.value, node.right.value)
                helper(node.right)

    helper(root)
    return graph


def average_shortest_path(root):
    """Return the average shortest path length of a discourse tree graph."""
    graph = tree_to_graph(root)
    return nx.average_shortest_path_length(graph)


def recover_list(token_list):
    """Recover surface text from tokenized subword pieces."""
    string = ""
    for token in token_list:
        if token[0] == "▁":
            string += " "
            string += token[1:]
        else:
            string += token
    return string


def sent2edu_map(sents, edus):
    """Map tokenized summary sentences to the EDU ids they cover."""
    sent2edu_dict = {}
    edus = [y.lower().strip() for y in edus]
    edu_idx = 0
    for idx, sent in enumerate(sents):
        sent = sent.lower().strip()
        sent2edu_dict[idx] = []
        cur_edu = edu_idx
        while cur_edu < len(edus):
            if (
                edus[cur_edu].strip() in sent.strip()
                or sent.strip() in edus[cur_edu].strip()
                or fuzz.partial_ratio(sent.strip(), edus[cur_edu].strip()) >= 90
            ):
                sent2edu_dict[idx].append(cur_edu + 1)
                cur_edu += 1
            else:
                if idx < len(sents) - 1 and (
                    edus[cur_edu].strip() in sents[idx - 1] + " " + sents[idx]
                    or fuzz.partial_ratio(sents[idx - 1] + " " + sents[idx], edus[cur_edu].strip()) >= 70
                ):
                    sent2edu_dict[idx].append(cur_edu + 1)
                    cur_edu += 1
                edu_idx = cur_edu
                break

    return sent2edu_dict


def max_height(root):
    """Return the maximum node depth of a binary tree."""
    if root is None:
        return 0
    left_height = max_height(root.left)
    right_height = max_height(root.right)
    return max(left_height, right_height) + 1


class TreeNode:
    """Binary discourse tree node with nuclearity and relation metadata."""

    def __init__(self, value, nuclearity, relation, left=None, right=None):
        self.value = value
        self.nuclearity = nuclearity
        self.relation = relation
        self.left = left
        self.right = right
        self.promotion_set = set()


def build_tree(preorder):
    """Build a discourse tree from a preorder list of span descriptors."""
    index = [0]

    def create_node(value, nuclearity, relation):
        return TreeNode(value, nuclearity, relation)

    def is_leaf(value):
        if "-" in value:
            start, end = value.split("-")
            return start == end
        return False

    def split_value(value):
        if ":" in value:
            return value.split(":")
        return value, None

    def build():
        value = preorder[index[0]][0]
        left_val, right_val = split_value(value)

        nuclearity = preorder[index[0]][1]
        relation = preorder[index[0]][2]
        index[0] += 1

        node = TreeNode(value, nuclearity, relation)
        if is_leaf(left_val):
            node.left = create_node(left_val, nuclearity.split("|")[0], relation.split("|")[0])
        else:
            node.left = build()
        
        if right_val:
            if is_leaf(right_val):
                node.right = create_node(right_val, nuclearity.split("|")[1], relation.split("|")[1])
            else:
                # Adjust for nodes that are explicitly split in the input
                if index[0] < len(preorder):
                    node.right = build()
        return node

    return build()


def translate_preorder(raw_str):
    """Translate raw parser output into the preorder tuples used by `build_tree`."""
    raw_str_lst = raw_str.split()
    preorder_list = []
    for node_str in raw_str_lst:
        nodes = node_str.split(",")
        node1 = nodes[0]
        node2 = nodes[1]
        l_node_start, l_relation, l_node_end = node1[1:].split(":")
        r_node_start, r_relation, r_node_end = node2[:-1].split(":")
        
        l_nuclearity, l_relation = l_relation.split("=")
        r_nuclerity, r_relation = r_relation.split("=")

        tuple_ = (
            f"{l_node_start}-{l_node_end}:{r_node_start}-{r_node_end}",
            f"{l_nuclearity}|{r_nuclerity}",
            f"{l_relation}|{r_relation}",
        )
        preorder_list.append(tuple_)
    return preorder_list


def print_relation_tree(node, depth=0):
    """Print the tree top-down with relation labels for debugging."""
    if not node:
        return
    indent = " " * depth * 2
    print(f"{indent}{node.value} --> {node.relation}")
    print_relation_tree(node.left, depth + 1)
    print_relation_tree(node.right, depth + 1)


def find_relation(node, query, depth=0):
    """Print matching relation metadata for a queried span."""
    if node is None:
        return
    if query == node.value:
        print(f"{node.value} || {node.relation} || {node.nuclearity} || {depth}")
    elif query in node.value:
        print(f"{node.value} || {node.relation} || {node.nuclearity} || {depth + 1}")

    find_relation(node.left, query, depth + 1)
    find_relation(node.right, query, depth + 1)


def find_node_and_compute(root, target):
    """Find a span node and return the node plus subtree size and height."""
    if root is None:
        return None, 0, 0

    if target in root.value.split(":") or target == root.value:
        if target == root.value.split(":")[1]:
            subtree_size = compute_size(root.right)
            subtree_height = compute_height(root.right)
        elif target == root.value.split(":")[0]:
            subtree_size = compute_size(root.left)
            subtree_height = compute_height(root.left)
        elif target == root.value:
            subtree_size = compute_size(root)
            subtree_height = compute_height(root)
        return root, subtree_size, subtree_height

    left_result = find_node_and_compute(root.left, target)
    if left_result[0] is not None:
        return left_result

    right_result = find_node_and_compute(root.right, target)
    return right_result


def compute_size(node):
    """Count the number of nodes in a subtree."""
    if node is None:
        return 0
    return 1 + compute_size(node.left) + compute_size(node.right)


def compute_height(node):
    """Return subtree height, using -1 for an empty tree."""
    if node is None:
        return -1
    return 1 + max(compute_height(node.left), compute_height(node.right))


def build_discourse_tree(raw_str):
    """Parse a raw discourse string and prepare its promotion annotations."""
    preorder_input = translate_preorder(raw_str)
    tree_root = build_tree(preorder_input)
    tree_depths = compute_height(tree_root) + 1
    update_promotion_sets(tree_root)
    return tree_root, tree_depths


def recover_sent(edus, level_parse):
    """Recover text chunks from EDU id ranges in a level-order parse."""
    text_segmentations = []
    for item in level_parse:
        try:
            chunk1, chunk2 = item.split(":")
            start_id, end_id = chunk1.split("-")
            text_segmentations.append("".join(edus[int(start_id) - 1 : int(end_id)]))

            start_id, end_id = chunk2.split("-")
            text_segmentations.append("".join(edus[int(start_id) - 1 : int(end_id)]))
        except ValueError:
            start_id, end_id = item.split("-")
            text_segmentations.append("".join(edus[int(start_id) - 1 : int(end_id)]))
    return text_segmentations


def level_order_traversal(root):
    """Return node spans grouped by tree depth, preserving recovered leaves."""
    if not root:
        return []

    queue = deque([root])
    level_order_values = []

    while queue:
        level_size = len(queue)
        current_level_values = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level_values.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        level_order_values.append(current_level_values)

    recovered_level_order = []
    recovered_level_order.append(level_order_values[0])
    for i, level_values in enumerate(level_order_values[1:], start=1):
        temp = level_values + [y for y in recovered_level_order[i - 1] if len(set(y.split("-"))) == 1]
        recovered_level_order.append(temp)

    return recovered_level_order



def calculate_penalties(node, current_penalty=0):
    """Assign satellite penalties to leaf EDUs along each root-to-leaf path."""
    if node is None:
        return []

    if node.left is None and node.right is None:
        return [(int(node.value.split("-")[0]), current_penalty)]

    results = []
    if node.left:
        additional_penalty = 1 if "satellite" in node.nuclearity.lower().split("|")[0] else 0
        results.extend(calculate_penalties(node.left, current_penalty + additional_penalty))
    if node.right:
        additional_penalty = 1 if "satellite" in node.nuclearity.lower().split("|")[1] else 0
        results.extend(calculate_penalties(node.right, current_penalty + additional_penalty))

    return results


def update_promotion_sets(node):
    """Populate promotion sets bottom-up based on nuclearity decisions."""
    if node is None:
        return set()

    left_promotion = update_promotion_sets(node.left)
    right_promotion = update_promotion_sets(node.right)

    if "nucleus" in node.nuclearity.lower() and "|" not in node.nuclearity.lower():
        node.promotion_set.update(left_promotion)
        node.promotion_set.update(right_promotion)
        node.promotion_set.add(node.value)
        return node.promotion_set.copy()
    elif "|" in node.nuclearity.lower():
        if "nucleus" in node.nuclearity.split("|")[0].lower():
            node.promotion_set.update(left_promotion)
            node.promotion_set.add(node.value.split(":")[0])
            return node.promotion_set.copy()
        elif "nucleus" in node.nuclearity.split("|")[1].lower():
            node.promotion_set.update(right_promotion)
            node.promotion_set.add(node.value.split(":")[1])
            return node.promotion_set.copy()


def print_raw_tree(node, depth=0):
    """Print a raw tree with relation labels in preorder layout."""
    if not node:
        return
    indent = " " * depth * 2
    print(f"{indent}{node.value} --> {node.relation}")
    print_raw_tree(node.left, depth + 1)
    print_raw_tree(node.right, depth + 1)


def print_tree(node, level=0):
    """Print the tree sideways together with each node's promotion set."""
    if node is not None:
        print_tree(node.right, level + 1)
        print(" " * 4 * level + "##>", node.value, node.promotion_set)
        print_tree(node.left, level + 1)


def is_in_promotion(list_span, target):
    """Check whether a promotion set contains a single-EDU target span."""
    for item in list_span:
        start_id, end_id = int(item.split("-")[0]), int(item.split("-")[1])
        if start_id == target == end_id:
            return True
    return False


def find_promption_depth(root, target, depth):
    """Return the deepest level where a target EDU remains promoted."""
    if root is None:
        return 0

    if is_in_promotion(root.promotion_set, target):
        return depth
    elif (root.left is not None and f"{target}-{target}" == root.left.value) or (
        root.right is not None and f"{target}-{target}" == root.right.value
    ):
        return depth - 1

    left_result = find_promption_depth(root.left, target, depth - 1)
    right_result = find_promption_depth(root.right, target, depth - 1)
    return max(left_result, right_result)


def find_upward_prompotion_score(root, target, score):
    """Count promoted ancestors encountered while walking upward from a target EDU."""
    if root is None:
        return score + 1
    if (root.left is not None and f"{target}-{target}" == root.left.value) or (
        root.right is not None and f"{target}-{target}" == root.right.value
    ):
        return score

    if is_in_promotion(root.promotion_set, target):
        score += 1
    left_result = find_upward_prompotion_score(root.left, target, score)
    right_result = find_upward_prompotion_score(root.right, target, score)
    return max(left_result, right_result)


def find_sent_upwardScore(tree_root, sent_span):
    """Return the maximum upward-promotion score for a sentence span."""
    start_id, end_id = int(sent_span.split("-")[0]), int(sent_span.split("-")[1])
    max_upward = 0
    for i in range(start_id, end_id + 1):
        cur_max = find_upward_prompotion_score(tree_root, i, 0)
        if max_upward < cur_max:
            max_upward = cur_max
    return max_upward


def find_sent_depthScore(tree_root, sent_span, _tree_height):
    """Return the maximum promotion depth score for a sentence span."""
    tree_height = max_height(tree_root) - 1

    start_id, end_id = int(sent_span.split("-")[0]), int(sent_span.split("-")[1])
    max_upward = 0
    for i in range(start_id, end_id + 1):
        cur_max = find_promption_depth(tree_root, i, tree_height)
        if max_upward < cur_max:
            max_upward = cur_max
    return max_upward


def find_sent_penality(tree_root, sent_span):
    """Return the maximum penalty assigned to any EDU in a sentence span."""
    edu_penalties = calculate_penalties(tree_root)
    edu_penality_dict = {x[0]: x[1] for x in edu_penalties}

    start_id, end_id = int(sent_span.split("-")[0]), int(sent_span.split("-")[1])
    max_penalty = 0
    for i in range(start_id, end_id + 1):
        if edu_penality_dict[i] > max_penalty:
            max_penalty = edu_penality_dict[i]
    return max_penalty


def compute_discourse_scores(raw_str, span, tree_root=None, tree_depths=None):
    """Compute depth, promotion, and penalty scores for a target span."""
    if tree_root is None or tree_depths is None:
        tree_root, tree_depths = build_discourse_tree(raw_str)

    max_upward_score = find_sent_upwardScore(tree_root, span)
    max_depth_score = find_sent_depthScore(tree_root, span, tree_depths)
    max_sent_penalty = find_sent_penality(tree_root, span)

    return {
        "max_upward_score": max_upward_score,
        "max_depth_score": max_depth_score,
        "max_sent_penalty": max_sent_penalty,
        "normalized_max_up_score": max_upward_score / tree_depths,
        "normalized_max_depth_score": max_depth_score / tree_depths,
        "normalized_max_sent_penalty": max_sent_penalty / tree_depths,
    }


def process_sent_score(str_):
    """Parse a serialized score list like `[0.1 0.2 0.3]` into floats."""
    str_list = str_.replace("[", "").replace("]", "").replace("\n", "").replace(",", " ").split()
    return [float(y.strip()) for y in str_list]


def reweight_score(score_list, row, alpha=1, depth_factor=1):
    """Apply discourse-aware reweighting to a sentence-level score list.

    The function maps summary sentences to EDUs, computes discourse-based depth
    and promotion signals for each aligned sentence span, and then rescales the
    original sentence scores before returning both the new and original means.
    """
    summ_segment = literal_eval(row["summ_segments"])
    if len(summ_segment) == 1:
        summ_segment = summ_segment[0]

    summ_parsetree = literal_eval(row["summ_tree_parsing"])
    if len(summ_parsetree) == 1:
        summ_parsetree = summ_parsetree[0][0]

    summ_sent_parse = literal_eval(row["summ_sents"])[0]

    if summ_parsetree == "NONE":
        return np.mean(score_list), np.mean(score_list), min(score_list), min(score_list), score_list
    edus = []
    for idx, end_id in enumerate(summ_segment):
        if idx == 0:
            edus.append(summ_sent_parse[: end_id + 1])
        else:
            edus.append(summ_sent_parse[summ_segment[idx - 1] + 1 : end_id + 1])
    edu_spans = [recover_list(item) for item in edus]
    real_summ = sent_tokenize(row["summary"])
    assert len(real_summ) == len(score_list)
    sent2edu_dict = sent2edu_map(real_summ, edu_spans)

    if list(sent2edu_dict.values()).count([]) >= 3:
        x2 = fix_alignment(
            remove_overlapping_values(
                align_sentences_and_segments_fuzzy(real_summ, edu_spans, threshold=70)
            )
        )
        for key in sent2edu_dict:
            if key not in x2:
                x2[key] = sent2edu_dict[key]
        sent2edu_dict = x2

    tree_root, tree_depths = build_discourse_tree(summ_parsetree)

    all_penalty_scores = []
    all_promote_scores = []
    all_depth_scores = []

    all_ono_penalty_scores = []

    for id1 in sent2edu_dict:
        non_factual_segments = sent2edu_dict[id1]
        if not non_factual_segments:
            all_penalty_scores.append(-1)
            all_promote_scores.append(-1)
            all_depth_scores.append(0)
            all_ono_penalty_scores.append(-1)
        else:
            segment_str = str(non_factual_segments[0]) + "-" + str(non_factual_segments[-1])

            max_scores = compute_discourse_scores(
                summ_parsetree,
                segment_str,
                tree_root=tree_root,
                tree_depths=tree_depths,
            )

            found_node, _, depth = find_node_and_compute(tree_root, segment_str)
            if found_node is None:
                start_id, end_id = segment_str.split("-")
                try:
                    depth = int(np.sqrt(int(end_id) - int(start_id)))
                except ValueError:
                    depth = 0

            all_depth_scores.append(depth)
            all_penalty_scores.append(max_scores["normalized_max_depth_score"])
            all_promote_scores.append(max_scores["normalized_max_up_score"])
            all_ono_penalty_scores.append(max_scores["normalized_max_sent_penalty"])

    assert len(all_depth_scores) == len(all_penalty_scores)

    mean_penalty = np.mean([y for y in all_penalty_scores if y != -1])
    mean_promote = np.mean([y for y in all_promote_scores if y != -1])
    mean_ono_penalty = np.mean([y for y in all_ono_penalty_scores if y != -1])

    all_penalty_scores = [y if y != -1 else mean_penalty for y in all_penalty_scores]
    all_promote_scores = [y if y != -1 else mean_promote for y in all_promote_scores]
    all_ono_penalty_scores = [y if y != -1 else mean_ono_penalty for y in all_ono_penalty_scores]

    updated_score = [
        alpha * score_list[i] ** (1 + mean_penalty - all_penalty_scores[i])
        if score_list[i] <= 1
        else score_list[i]
        for i in range(len(score_list))
    ]
    updated_complexity_score = [
        updated_score[i] ** (1 + all_depth_scores[i] * depth_factor)
        for i in range(len(updated_score))
    ]

    return np.mean(updated_complexity_score), np.mean(score_list), min(updated_score), min(score_list), updated_complexity_score
