# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

import os
import copy
import distance
import numpy as np
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def parallel_process_notqdm(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in array[front_num:]]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in as_completed(futures):
            pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            try:
                colspan = int(node.attrib.get('colspan', '1'))
            except:
                colspan = 1
            try:
                rowspan = int(node.attrib.get('rowspan', '1'))
            except:
                rowspan = 1
            new_node = TableTree(node.tag,
                                 colspan,
                                 rowspan,
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            treedistance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(treedistance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = [item if isinstance(item, float) else 0.0 for item in scores]
        mean_score = round(np.mean(scores), 4)
        scores = dict(zip(samples, scores))
        return mean_score, scores
    
    def batch_evaluate_html(self, pred_htmls, true_htmls):
        """Computes TEDS score between the prediction and the ground truth of
        a batch of samples
        """
        if self.n_jobs == 1:
            scores = [
                self.evaluate(pred_html, true_html)
                for (pred_html, true_html) in tqdm(zip(pred_htmls, true_htmls))
            ]
        else:
            inputs = [
                {"pred": pred_html, "true": true_html}
                for (pred_html, true_html) in zip(pred_htmls, true_htmls)
            ]

            scores = parallel_process_notqdm(
                inputs, self.evaluate, use_kwargs=True, n_jobs=1, front_num=1
            )
        scores = [item if isinstance(item, float) else 0.0 for item in scores]
        return scores


class OEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.config = CustomOEDSConfig
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = OTSLTree(node.tag, node.attrib.get('structure', 'C'), cell, *deque())
        else:
            new_node = OTSLTree(node.tag, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node
    
    def html_to_otsl(self, html_table):
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        if not table:
            return ""

        rows = table.find_all('tr')

        # 创建一个二维数组来存储展开后的表格
        max_cols = 0
        prod_rowspan_list = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            curr_cols = 0
            for cell in cells:
                curr_cols += int(cell.get('colspan', 1))
            new_list = []
            for prod_rowspan in prod_rowspan_list:
                curr_cols += 1
                prod_rowspan -= 1
                if prod_rowspan > 0:
                    new_list.append(prod_rowspan)
            for cell in cells:
                if int(cell.get("rowspan", 1)) > 1:
                    new_list.append(int(cell.get("rowspan", 1)) - 1)
            prod_rowspan_list = new_list
            max_cols = max(max_cols, curr_cols)

        grid = [[None] * max_cols for _ in range(len(rows))]
        cell_contents = [['' for _ in range(max_cols)] for _ in range(len(rows))]

        # 填充表格
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            col_idx = 0

            for cell in cells:
                # 找到下一个空位置
                while col_idx < max_cols and grid[i][col_idx] is not None:
                    col_idx += 1

                if col_idx >= max_cols:
                    break
                    
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))

                # 获取单元格内容
                content = cell.get_text(strip=True)

                
                if rowspan > 1 or colspan > 1:
                    grid[i][col_idx] = "M" # 标记合并主单元格为M
                else:
                    grid[i][col_idx] = "C" # 标记普通单元格为C
                cell_contents[i][col_idx] = content

                # 填充当前单元格及其跨行跨列区域
                for r in range(i, i + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r >= len(grid) or c >= max_cols:
                            continue

                        if r == i and c == col_idx:
                            continue  # 跳过主单元格

                        if r == i:  # 同一行，左合并
                            grid[r][c] = "L"  # L
                        elif c == col_idx:  # 同一列，上合并
                            grid[r][c] = "U"  # U
                        else:  # 交叉合并
                            grid[r][c] = "X"  # X
                
                col_idx += colspan

        otsl = ["<html><body><table border=\"1\" >"]
        for i in range(len(grid)):
            otsl.append("<tr>\n")
            for j in range(len(grid[i])):
                if grid[i][j] == "M" or grid[i][j] == "C":  # M
                    content = cell_contents[i][j]
                    otsl.append(f"<td structure=\"{grid[i][j]}\">{content}</td>")
                elif grid[i][j] == "L" or grid[i][j] == "U" or grid[i][j] == "X":  # L
                    otsl.append(f"<td structure=\"{grid[i][j]}\"></td>")
                else:
                    otsl.append(f"<td structure=\"A\"></td>")
            otsl.append("</tr>\n")
        otsl.append("</table></body></html>")
        return "".join(otsl)

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        pred, true = self.html_to_otsl(pred), self.html_to_otsl(true)
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)

            treedistance = APTED(tree_pred, tree_true, self.config()).compute_edit_distance()
            return 1.0 - (float(treedistance) / n_nodes)
        else:
            return 0.0

    def __call__(self, pred, true):
        return self.evaluate(pred, true)

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        # if self.n_jobs == 1:
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        # else:
        #     inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
        #     scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores
    
    def batch_evaluate_html(self, pred_htmls, true_htmls):
        """Computes TEDS score between the prediction and the ground truth of
        a batch of samples
        """
        if self.n_jobs == 1:
            scores = [
                self.evaluate(pred_html, true_html)
                for (pred_html, true_html) in zip(pred_htmls, true_htmls)
            ]
        else:
            inputs = [
                {"pred": pred_html, "true": true_html}
                for (pred_html, true_html) in zip(pred_htmls, true_htmls)
            ]

            scores = parallel_process_notqdm(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1
            )
        scores = [item if isinstance(item, float) else 0.0 for item in scores]
        return scores


class OTSLTree(Tree):
    def __init__(self, tag, structure=None, content=None, *children):
        self.tag = tag
        self.structure = structure
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "structure": %s, "text": %s' % \
                     (self.tag, self.structure, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomOEDSConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.structure != node2.structure):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


if __name__ == "__main__":
    a = """
    <table>
    <tr><td rowspan="4">12</td><td>1</td></tr>
    <tr><td>1</td></tr>
    <tr><td>12</td></tr>
    <tr><td>12345</td></tr>
    </table>
    """
    b = """
    <table>
    <tr><td rowspan="4">12</td><td>1</td></tr>
    <tr><td>1</td></tr>
    <tr><td>12</td></tr>
    <tr><td>12345</td></tr>
    <tr></tr>
    </table>
    """
    oeds, s_oeds = OEDS(), OEDS(structure_only=True)
    oed_s, oed_s = oeds(a, b), s_oeds(a, b)
    print(oed_s, oed_s)

    teds, s_teds = TEDS(), TEDS(structure_only=True)
    a = "<html><body>" + a + "</body></html>"
    b = "<html><body>" + b + "</body></html>"
    ted_s, ted_s = teds(a, b), s_teds(a, b)
    print(ted_s, ted_s)