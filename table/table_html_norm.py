import re
import html
import uuid
import subprocess
import unicodedata
from bs4 import BeautifulSoup

    
def normalized_html_table(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()
            tbody_tags = soup.find_all('tbody')
            for tbody in tbody_tags:
                tbody.unwrap()
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            
            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            # table_res_no_space = re.sub(' (style=".*?")',"",table_res_no_space)
            # table_res_no_space = re.sub(r'[ ]', " ", table_res_no_space)
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'
            # table_flow.append(table_res)
            # table_flow_no_space.append(table_res_no_space)

        return table_res, table_res_no_space
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    
    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text.replace('> ', '>').replace(" </td>", "</td>")