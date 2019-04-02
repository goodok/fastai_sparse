plantuml ./classes.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py classes.svg --color_hrefs '#0000a0'
# rsvg-convert -o classes.png classes.svg -w 800 --keep-aspect-ratio

plantuml ./classes_short.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py classes_short.svg --color_hrefs '#0000a0'
# rsvg-convert -o classes_short.png classes_short.svg -w 800 --keep-aspect-ratio
