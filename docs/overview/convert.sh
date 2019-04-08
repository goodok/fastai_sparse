plantuml ./classes.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py classes.svg --links_color '#0000f0' --ignore_links="3D augmentation"
#rsvg-convert -o classes.png classes.svg -w 800 --keep-aspect-ratio

plantuml ./classes_short.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py classes_short.svg --links_color '#0000f0' --ignore_links="3D augmentation"
#rsvg-convert -o classes_short.png classes_short.svg -w 800 --keep-aspect-ratio

plantuml ./augmentation.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py augmentation.svg --links_color '#0000f0'
#rsvg-convert -o augmentation.png augmentation.svg -w 800 --keep-aspect-ratio

