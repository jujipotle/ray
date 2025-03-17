import time
import threading
import heapq
from collections import deque, defaultdict

def current_millis():
    return int(time.time() * 1000)

def shared_prefix_count(a: str, b: str) -> int:
    count = 0
    for ac, bc in zip(a, b):
        if ac == bc:
            count += 1
        else:
            break
    return count

def slice_by_chars(s: str, start: int, end: int) -> str:
    return s[start:end]

class Node:
    def __init__(self, text=""):
        self.children = {}  # mapping from first character to child Node
        self.text = text
        self.tenant_last_access_time = {}  # tenant -> timestamp (ms)
        self.parent = None
        self.lock = threading.RLock()

class Tree:
    def __init__(self):
        self.root = Node("")
        self.tenant_char_count = {}  # tenant -> total char count
        self.lock = threading.RLock()

    def insert(self, text: str, tenant: str):
        with self.lock:
            curr_idx = 0
            timestamp_ms = current_millis()
            # update root tenant access
            self.root.tenant_last_access_time[tenant] = timestamp_ms
            if tenant not in self.tenant_char_count:
                self.tenant_char_count[tenant] = 0

            prev = self.root
            text_count = len(text)
            while curr_idx < text_count:
                first_char = text[curr_idx]
                curr = prev
                if first_char not in curr.children:
                    # no match: create new node with remaining text
                    curr_text = text[curr_idx:]
                    curr_text_count = len(curr_text)
                    new_node = Node(curr_text)
                    new_node.parent = curr
                    new_node.tenant_last_access_time[tenant] = timestamp_ms
                    curr.children[first_char] = new_node
                    self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + curr_text_count
                    prev = new_node
                    curr_idx = text_count
                else:
                    matched_node = curr.children[first_char]
                    matched_node_text = matched_node.text
                    matched_node_text_count = len(matched_node_text)
                    curr_text = text[curr_idx:]
                    shared_count = shared_prefix_count(matched_node_text, curr_text)
                    if shared_count < matched_node_text_count:
                        # split the matched node
                        matched_text = matched_node_text[:shared_count]
                        contracted_text = matched_node_text[shared_count:]
                        matched_text_count = len(matched_text)
                        new_node = Node(matched_text)
                        new_node.parent = curr
                        # copy existing tenant timestamps
                        new_node.tenant_last_access_time = matched_node.tenant_last_access_time.copy()
                        if contracted_text:
                            new_node.children[contracted_text[0]] = matched_node
                        curr.children[first_char] = new_node
                        matched_node.text = contracted_text
                        matched_node.parent = new_node
                        prev = new_node
                        if tenant not in new_node.tenant_last_access_time:
                            self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + matched_text_count
                        new_node.tenant_last_access_time[tenant] = timestamp_ms
                        curr_idx += shared_count
                    else:
                        prev = matched_node
                        if tenant not in matched_node.tenant_last_access_time:
                            self.tenant_char_count[tenant] = self.tenant_char_count.get(tenant, 0) + matched_node_text_count
                        matched_node.tenant_last_access_time[tenant] = timestamp_ms
                        curr_idx += shared_count

    def prefix_match(self, text: str):
        with self.lock:
            curr_idx = 0
            prev = self.root
            text_count = len(text)
            while curr_idx < text_count:
                first_char = text[curr_idx]
                curr_text = text[curr_idx:]
                if first_char in prev.children:
                    matched_node = prev.children[first_char]
                    shared_count = shared_prefix_count(matched_node.text, curr_text)
                    if shared_count == len(matched_node.text):
                        curr_idx += shared_count
                        prev = matched_node
                    else:
                        curr_idx += shared_count
                        prev = matched_node
                        break
                else:
                    break
            # update timestamps along the path
            tenant = next(iter(prev.tenant_last_access_time), "empty")
            if tenant != "empty":
                timestamp_ms = current_millis()
                node = prev
                while node:
                    node.tenant_last_access_time[tenant] = timestamp_ms
                    node = node.parent
            return text[:curr_idx], tenant

    def prefix_match_tenant(self, text: str, tenant: str):
        with self.lock:
            curr_idx = 0
            prev = self.root
            text_count = len(text)
            while curr_idx < text_count:
                first_char = text[curr_idx]
                curr_text = text[curr_idx:]
                if first_char in prev.children:
                    matched_node = prev.children[first_char]
                    if tenant not in matched_node.tenant_last_access_time:
                        break
                    shared_count = shared_prefix_count(matched_node.text, curr_text)
                    if shared_count == len(matched_node.text):
                        curr_idx += shared_count
                        prev = matched_node
                    else:
                        curr_idx += shared_count
                        prev = matched_node
                        break
                else:
                    break
            if tenant in prev.tenant_last_access_time:
                timestamp_ms = current_millis()
                node = prev
                while node:
                    node.tenant_last_access_time[tenant] = timestamp_ms
                    node = node.parent
            return text[:curr_idx]

    @staticmethod
    def leaf_of(node: Node):
        candidates = {tenant: True for tenant in node.tenant_last_access_time.keys()}
        for child in node.children.values():
            for tenant in child.tenant_last_access_time.keys():
                candidates[tenant] = False
        return [tenant for tenant, is_leaf in candidates.items() if is_leaf]

    def evict_tenant_by_size(self, max_size: int):
        with self.lock:
            stack = [self.root]
            heap = []
            while stack:
                curr = stack.pop()
                for child in curr.children.values():
                    stack.append(child)
                for tenant in Tree.leaf_of(curr):
                    if tenant in curr.tenant_last_access_time:
                        timestamp = curr.tenant_last_access_time[tenant]
                        heapq.heappush(heap, (timestamp, tenant, curr))
            print("Before eviction - Used size per tenant:")
            for tenant, size in self.tenant_char_count.items():
                print(f"Tenant: {tenant}, Size: {size}")

            while heap:
                timestamp, tenant, node = heapq.heappop(heap)
                if self.tenant_char_count.get(tenant, 0) <= max_size:
                    continue
                if tenant in node.tenant_last_access_time:
                    node_text_len = len(node.text)
                    self.tenant_char_count[tenant] = max(self.tenant_char_count.get(tenant, 0) - node_text_len, 0)
                if tenant in node.tenant_last_access_time:
                    del node.tenant_last_access_time[tenant]
                if not node.children and not node.tenant_last_access_time:
                    if node.parent:
                        parent = node.parent
                        key = node.text[0] if node.text else None
                        if key and key in parent.children and parent.children[key] == node:
                            del parent.children[key]
                if node.parent and tenant in Tree.leaf_of(node.parent):
                    parent = node.parent
                    if tenant in parent.tenant_last_access_time:
                        ts = parent.tenant_last_access_time[tenant]
                        heapq.heappush(heap, (ts, tenant, parent))
            print("After eviction - Used size per tenant:")
            for tenant, size in self.tenant_char_count.items():
                print(f"Tenant: {tenant}, Size: {size}")

    def remove_tenant(self, tenant: str):
        with self.lock:
            stack = [self.root]
            queue = deque()
            while stack:
                curr = stack.pop()
                for child in curr.children.values():
                    stack.append(child)
                if tenant in Tree.leaf_of(curr):
                    queue.append(curr)
            while queue:
                curr = queue.popleft()
                if tenant in curr.tenant_last_access_time:
                    del curr.tenant_last_access_time[tenant]
                if not curr.children and not curr.tenant_last_access_time:
                    if curr.parent:
                        parent = curr.parent
                        key = curr.text[0] if curr.text else None
                        if key and key in parent.children and parent.children[key] == curr:
                            del parent.children[key]
                if curr.parent and tenant in Tree.leaf_of(curr.parent):
                    queue.append(curr.parent)
            if tenant in self.tenant_char_count:
                del self.tenant_char_count[tenant]

    def get_tenant_char_count(self):
        with self.lock:
            return dict(self.tenant_char_count)

    def get_smallest_tenant(self):
        with self.lock:
            if not self.tenant_char_count:
                return "empty"
            min_tenant = None
            min_count = float('inf')
            for tenant, count in self.tenant_char_count.items():
                if count < min_count:
                    min_count = count
                    min_tenant = tenant
            return min_tenant if min_tenant is not None else "empty"

    def get_used_size_per_tenant(self):
        with self.lock:
            used_size = defaultdict(int)
            stack = [self.root]
            while stack:
                curr = stack.pop()
                text_len = len(curr.text)
                for tenant in curr.tenant_last_access_time:
                    used_size[tenant] += text_len
                for child in curr.children.values():
                    stack.append(child)
            return dict(used_size)

    def node_to_string(self, node: Node, prefix: str, is_last: bool) -> str:
        result = prefix
        result += "└── " if is_last else "├── "
        result += f"'{node.text}' ["
        tenant_info = []
        for tenant, timestamp in node.tenant_last_access_time.items():
            t = time.localtime(timestamp / 1000)
            millis = timestamp % 1000
            tenant_info.append(f"{tenant} | {time.strftime('%H:%M:%S', t)}.{millis:03d}")
        result += ", ".join(tenant_info)
        result += "]\n"
        children = list(node.children.values())
        child_count = len(children)
        for i, child in enumerate(children):
            is_last_child = (i == child_count - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            result += self.node_to_string(child, new_prefix, is_last_child)
        return result

    def pretty_print(self):
        with self.lock:
            if not self.root.children:
                return
            result = ""
            children = list(self.root.children.values())
            child_count = len(children)
            for i, child in enumerate(children):
                is_last = (i == child_count - 1)
                result += self.node_to_string(child, "", is_last)
            print(result)