from bs4 import BeautifulSoup



def html_to_otsl(html_table):
    html_table = html_table.replace("<br>", "\n")
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    if not table:
        return ""

    rows = table.find_all('tr')

    # 创建一个二维数组来存储展开后的表格
    max_cols = 0
    for row in rows:
        cells = row.find_all(['td', 'th'])
        curr_cols = 0
        for cell in cells:
            curr_cols += int(cell.get('colspan', 1))
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
            try:    
                rowspan = int(cell.get('rowspan', 1))
            except:
                rowspan = 1
            try:
                colspan = int(cell.get('colspan', 1))
            except:
                colspan = 1

            # 获取单元格内容
            content = cell.get_text(strip=True)

            
            if content != "":
                grid[i][col_idx] = "<fcel>" # 标记合并主单元格为M
            else:
                grid[i][col_idx] = "<ecel>" # 标记普通单元格为C
            cell_contents[i][col_idx] = content

            # 填充当前单元格及其跨行跨列区域
            for r in range(i, i + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    if r >= len(grid) or c >= max_cols:
                        continue

                    if r == i and c == col_idx:
                        continue  # 跳过主单元格

                    if r == i:  # 同一行，左合并
                        grid[r][c] = "<lcel>"  # L
                    elif c == col_idx:  # 同一列，上合并
                        grid[r][c] = "<ucel>"  # U
                    else:  # 交叉合并
                        grid[r][c] = "<xcel>"  # X
            
            col_idx += colspan

    otsl = [""]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            otsl_tag = grid[i][j]
            if not otsl_tag:
                otsl_tag = "<ecel>"
            if i == 0 and otsl_tag == "<ucel>":
                otsl_tag = "<ecel>"
            if j == 0 and otsl_tag == "<lcel>":
                otsl_tag = "<ecel>"
            content = cell_contents[i][j]
            otsl.append(otsl_tag + content.strip())
        otsl.append("<nl>\n")
    return "".join(otsl)


if __name__ == "__main__":
    html_table = """
    <table>
        <tr>
            <td colspan="2">Header 1</td>
            <td>Header 2</td>
        </tr>
        <tr>
            <td>Row 1, Col 1</td>
            <td>Row 1 > Col 2</td>
        </tr>
    </table>
    """
    otsl = html_to_otsl(html_table)
    print(otsl)