import os
import ast
import math
from collections import Counter, defaultdict
import numpy as np

MATCH_SCORE=3
GAP_COST=2

def compute_tf(ast_tree, all_nodes):
    """Compute term frequency for an AST given a list of all node types."""
    nodes = [type(node).__name__ for node in ast.walk(ast_tree)]
    node_count = Counter(nodes)
    total_nodes = sum(node_count.values())
    # Return a list of term frequencies in the same order as all_nodes
    return [node_count.get(node, 0) / total_nodes for node in all_nodes]


def compute_df(ast_trees):
    """Compute document frequency for all nodes given a group of ASTs."""
    df_counter = Counter()
    for ast_tree in ast_trees:
        nodes = set(type(node).__name__ for node in ast.walk(ast_tree))
        df_counter.update(nodes)
    return df_counter

def compute_idf(df, total_documents):
    return {node: math.log((total_documents + 1) / (frequency + 1)) + 1 for node, frequency in df.items()}


def compute_tfidf(ast_trees):
    """Compute TF-IDF for a collection of ASTs and return aligned vectors."""
    df = compute_df(ast_trees)
    idf = compute_idf(df, len(ast_trees))
    all_nodes = list(df.keys())
    tfidf_trees = []
    
    for ast_tree in ast_trees:
        tf = compute_tf(ast_tree, all_nodes)
        # Use the index to align tf with idf values
        tfidf = np.array([tf_val * idf[node] for node, tf_val in zip(all_nodes, tf)])
        tfidf_trees.append(tfidf)
    
    return tfidf_trees, all_nodes, idf

def compute_tfidf_ood(new_tree, all_nodes, existing_idf):
    """Compute TF-IDF for a new AST using existing IDF values."""
    tf_new_tree = compute_tf(new_tree, all_nodes)
    tfidf = np.array([tf_val * existing_idf[node] for node, tf_val in zip(all_nodes, tf_new_tree)])
    return tfidf

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def dfs_traversal(tree):
    """Perform a DFS traversal on an AST and return a list of node types."""
    nodes = []
    def visit(node):
        nodes.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    return nodes

def set_of_children(node, **kwargs):
    """Helper function for getting all the nodes in a subtree"""
    return set((type(node).__name__, )).union(set().union(*[set_of_children(child) for child in ast.iter_child_nodes(node)]))

def tree_edit_distance_with_operations(node1, node2):
    # Base cases
    if not node1 and not node2:
        return set()
    if not node1:
        return set_of_children(node2, annotate_fields=False)
    if not node2:
        return set_of_children(node1, annotate_fields=False)

    # Check if nodes are of same type
    if type(node1) != type(node2):
        return set_of_children(node2, annotate_fields=False).union(tree_edit_distance_with_operations(node1, None)) # delete faulty subtree, insert correct one

    else:
        children1 = list(ast.iter_child_nodes(node1))
        children2 = list(ast.iter_child_nodes(node2))

        # Get the cost and operations for matching children of both nodes
        operations = set()
        for c1, c2 in zip(children1, children2):
            ops = tree_edit_distance_with_operations(c1, c2)
            operations = operations.union(ops)

        # Extra children in either of the trees
        for extra_child in children1[len(children2):]:
            ops = tree_edit_distance_with_operations(extra_child, None)
            operations = operations.union(ops)
        for extra_child in children2[len(children1):]:
            ops = tree_edit_distance_with_operations(None, extra_child)
            operations = operations.union(ops)

        return operations

def print_ast(node, indent=0):
    """
    Recursively print an AST node.
    """
    # Print the node type and any additional information (e.g., its name if it's a function or variable)
    if isinstance(node, ast.FunctionDef):
        print('  ' * indent + f'FunctionDef(name={node.name})')
    elif isinstance(node, ast.Name):
        print('  ' * indent + f'Name(id={node.id})')
    else:
        print('  ' * indent + str(type(node).__name__))

    # For each field (like 'body' for functions, 'value' for assignments, etc.)
    for field in node._fields:
        value = getattr(node, field, None)
        if isinstance(value, list):
            for item in value:
                print_ast(item, indent+1)
        elif isinstance(value, ast.AST):
            print_ast(value, indent+1)

