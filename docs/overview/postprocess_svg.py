#!/usr/bin/env python
import lxml
from lxml import etree
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('input', metavar='input_file', type=str,
                    help='Input SVG file')
parser.add_argument('-o', '--output', dest='output', type=str, default=None,
                    help='Output SVG file')

parser.add_argument('--links_color', default=None, type=str,
                    help='Colorize link by color')

parser.add_argument('--ignore_links', default=None, type=str,
                    help='List of packages and class to be ignored from rearranging. Separated with comas.')


args = parser.parse_args()


fn_input = args.input
fn_output = args.output
if fn_output is None:
    fn_output = fn_input

# links_color = 'blue'
links_color = args.links_color

ignore_links = args.ignore_links
if ignore_links is not None:
    ignore_links = ignore_links.split(',')
else:
    ignore_links = []


def Main():

    with open(fn_input) as f:

        parser = etree.XMLParser(remove_blank_text=True)
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, strip_cdata=False)

        # etree.parse(f)
        doc = etree.parse(f, parser)

    root = doc.getroot()

    rearange_links(root)

    # pretify and save
    # print(etree.tostring(doc, pretty_print=True))
    doc.write(fn_output, pretty_print=True)


def rearange_links(root):
    """
    Replace:

    <!--cluster torch-->
    <a href=''>
      <polygon ... />
      <line ... />
      <text ..>xxx</text>
    </a>

    To:

    <!--cluster torch-->
    <polygon ... />
    <line ... />
    <a href=''>
      <text ..>xxx</text>
    </a>

    """
    ns_svg = root.nsmap[None]

    anchors = root.findall('.//a', namespaces=root.nsmap)

    for a in anchors:
        a.attrib['style'] = 'cursor: hand;'
        comment = a.getprevious()
        if isinstance(comment, lxml.etree._Comment):
            if comment.text.startswith('cluster '):
                claster_name = ''.join(comment.text.split('cluster ')[1:])
                if claster_name not in ignore_links:
                    g = etree.Element('g')
                    a.addprevious(g)
                    for c in a.getchildren():
                        if c.tag != "{" + ns_svg + "}text":
                            g.append(c)
                        elif links_color is not None:
                            c.attrib['fill'] = links_color

                    a.attrib['target'] = '_blank'
                    g.append(a)
            elif comment.text.startswith('class '):
                class_name = ''.join(comment.text.split('class ')[1:])
                if class_name not in ignore_links:
                    g = etree.Element('g')
                    a.addprevious(g)
                    for c in a.getchildren():
                        if c.tag != "{" + ns_svg + "}text":
                            g.append(c)
                        elif c.text != class_name:
                            g.append(c)
                        elif links_color is not None:
                            c.attrib['fill'] = links_color

                    a.attrib['target'] = '_blank'
                    g.append(a)


Main()
