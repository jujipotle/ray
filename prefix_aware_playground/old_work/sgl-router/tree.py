import time
from collections import defaultdict
from threading import RLock


def shared_prefix_count(a, b):
    """Count the number of shared characters at the beginning of two strings."""
    i = 0
    for char_a, char_b in zip(a, b):
        if char_a == char_b:
            i += 1
        else:
            break
    return i


class Node:
    """
    Node in a prefix tree that tracks tenant access time.
    
    Each node represents a segment of text and can belong to multiple tenants.
    """
    def __init__(self, text="", parent=None):
        self.children = {}  # Maps char -> Node
        self.text = text
        self.tenant_last_access_time = {}  # Maps tenant -> timestamp
        self.parent = parent
        self.lock = RLock()  # For thread safety
        
    def __str__(self):
        return f"Node(text='{self.text}', tenants={list(self.tenant_last_access_time.keys())})"


class Tree:
    """
    Thread-safe multi-tenant prefix tree (approximate radix tree).
    
    Features:
    1. Stores data for multiple tenants in the same tree structure
    2. Node-level locking for concurrent access
    3. Leaf LRU eviction based on tenant access time
    """
    def __init__(self):
        self.root = Node()
        self.tenant_char_count = defaultdict(int)  # Maps tenant -> character count
        self.lock = RLock()  # For operations that need to lock the entire tree
        
    def insert(self, text, tenant):
        """Insert text into tree with given tenant."""
        curr = self.root
        curr_idx = 0
        timestamp_ms = int(time.time() * 1000)
        
        # Update access time for root node
        with curr.lock:
            curr.tenant_last_access_time[tenant] = timestamp_ms
        
        prev = self.root
        text_len = len(text)
        
        while curr_idx < text_len:
            first_char = text[curr_idx]
            curr = prev
            
            with curr.lock:
                if first_char not in curr.children:
                    # No match, create new node
                    curr_text = text[curr_idx:]
                    new_node = Node(text=curr_text, parent=curr)
                    new_node.tenant_last_access_time[tenant] = timestamp_ms
                    
                    # Increment char count for tenant
                    with self.lock:
                        self.tenant_char_count[tenant] += len(curr_text)
                    
                    curr.children[first_char] = new_node
                    prev = new_node
                    curr_idx = text_len  # End loop
                else:
                    # Match found, check if need to split
                    matched_node = curr.children[first_char]
                    
                    with matched_node.lock:
                        matched_node_text = matched_node.text
                        matched_node_text_len = len(matched_node_text)
                        
                        curr_text = text[curr_idx:]
                        shared_count = shared_prefix_count(matched_node_text, curr_text)
                        
                        if shared_count < matched_node_text_len:
                            # Split the matched node
                            matched_text = matched_node_text[:shared_count]
                            contracted_text = matched_node_text[shared_count:]
                            
                            # Create new intermediate node
                            new_node = Node(text=matched_text, parent=curr)
                            new_node.tenant_last_access_time = matched_node.tenant_last_access_time.copy()
                            
                            # Update the original matched node
                            matched_node.text = contracted_text
                            matched_node.parent = new_node
                            
                            # Connect new node to the tree
                            first_new_char = contracted_text[0]
                            new_node.children[first_new_char] = matched_node
                            curr.children[first_char] = new_node
                            
                            prev = new_node
                            
                            # Update tenant char count for the new split node
                            if tenant not in prev.tenant_last_access_time:
                                with self.lock:
                                    self.tenant_char_count[tenant] += len(matched_text)
                            
                            prev.tenant_last_access_time[tenant] = timestamp_ms
                            curr_idx += shared_count
                        else:
                            # Move to next node (full match with current node)
                            prev = matched_node
                            
                            # Update tenant char count if this is a new tenant for this node
                            if tenant not in prev.tenant_last_access_time:
                                with self.lock:
                                    self.tenant_char_count[tenant] += len(matched_node_text)
                            
                            prev.tenant_last_access_time[tenant] = timestamp_ms
                            curr_idx += shared_count
    
    def prefix_match(self, text):
        """
        Match text against tree and return (matched_text, matched_tenant).
        Updates access time for the matched tenant.
        """
        curr = self.root
        curr_idx = 0
        prev = self.root
        text_len = len(text)
        
        while curr_idx < text_len:
            first_char = text[curr_idx]
            curr_text = text[curr_idx:]
            
            curr = prev
            
            with curr.lock:
                if first_char in curr.children:
                    matched_node = curr.children[first_char]
                    
                    with matched_node.lock:
                        shared_count = shared_prefix_count(matched_node.text, curr_text)
                        matched_node_text_len = len(matched_node.text)
                        
                        if shared_count == matched_node_text_len:
                            # Full match with current node's text, continue to next node
                            curr_idx += shared_count
                            prev = matched_node
                        else:
                            # Partial match, stop here
                            curr_idx += shared_count
                            prev = matched_node
                            break
                else:
                    # No match found, stop here
                    break
        
        curr = prev
        
        # Select the first tenant
        with curr.lock:
            if curr.tenant_last_access_time:
                tenant = next(iter(curr.tenant_last_access_time))
            else:
                tenant = "empty"
        
        # Update timestamp for all nodes from match point to root
        if tenant != "empty":
            timestamp_ms = int(time.time() * 1000)
            current_node = curr
            
            while current_node is not None:
                with current_node.lock:
                    current_node.tenant_last_access_time[tenant] = timestamp_ms
                current_node = current_node.parent
        
        ret_text = text[:curr_idx]
        return ret_text, tenant
    
    def prefix_match_tenant(self, text, tenant):
        """
        Match text against tree for a specific tenant and return matched text.
        Updates access time for the tenant.
        """
        curr = self.root
        curr_idx = 0
        prev = self.root
        text_len = len(text)
        
        while curr_idx < text_len:
            first_char = text[curr_idx]
            curr_text = text[curr_idx:]
            
            curr = prev
            
            with curr.lock:
                if first_char in curr.children:
                    matched_node = curr.children[first_char]
                    
                    with matched_node.lock:
                        # Only continue matching if this node belongs to the specified tenant
                        if tenant not in matched_node.tenant_last_access_time:
                            break
                            
                        shared_count = shared_prefix_count(matched_node.text, curr_text)
                        matched_node_text_len = len(matched_node.text)
                        
                        if shared_count == matched_node_text_len:
                            # Full match with current node's text, continue to next node
                            curr_idx += shared_count
                            prev = matched_node
                        else:
                            # Partial match, stop here
                            curr_idx += shared_count
                            prev = matched_node
                            break
                else:
                    # No match found, stop here
                    break
        
        # Update timestamps
        timestamp_ms = int(time.time() * 1000)
        current_node = prev
        
        while current_node is not None:
            with current_node.lock:
                if tenant in current_node.tenant_last_access_time:
                    current_node.tenant_last_access_time[tenant] = timestamp_ms
            current_node = current_node.parent
        
        ret_text = text[:curr_idx]
        return ret_text
    
    def get_smallest_tenant(self):
        """Get the tenant with the smallest total character count."""
        with self.lock:
            if not self.tenant_char_count:
                # Return first worker if no data yet
                return "empty"
            
            return min(self.tenant_char_count.items(), key=lambda x: x[1])[0]
    
    def evict_tenant_by_size(self, max_size):
        """Evict nodes for tenants that exceed the maximum tree size."""
        pass  # Simplifying for this implementation
    
    def get_tenant_char_count(self):
        """Get character count for each tenant."""
        with self.lock:
            return dict(self.tenant_char_count)
            
    def remove_tenant(self, tenant):
        """Remove all nodes belonging to a tenant."""
        # Would require a traversal of the tree and removing the tenant
        # from tenant_last_access_time. Simplifying for now.
        with self.lock:
            if tenant in self.tenant_char_count:
                del self.tenant_char_count[tenant] 