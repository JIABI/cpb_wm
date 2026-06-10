cd ~/PyCharmMiscProject/nms
 
python3 << 'EOF'

from pathlib import Path

import re

path = Path("src/nms/wm/minimal_dreamer.py")

text = path.read_text()
 
# 匹配 "values = self._critic(img_feats_t.reshape(B * H, -1)).reshape(B, H)"

# 改成 detach 版本

old = "values = self._critic(img_feats_t.reshape(B * H, -1)).reshape(B, H)"

new = "values = self._critic(img_feats_t.detach().reshape(B * H, -1)).reshape(B, H)  # detach to keep critic grads out of actor graph"
 
if old not in text:

    print("ERROR: target line not found")

    print("Run: grep -n 'self._critic(img_feats_t' src/nms/wm/minimal_dreamer.py")

else:

    path.write_text(text.replace(old, new))

    print("PATCHED: values now uses img_feats_t.detach()")

EOF
 