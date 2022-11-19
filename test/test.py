import requests

thingy = '[[1633514924089,-0.96875,-0.021484375,0.27246094,-0.25,-0.125,-0.0625,0.0,0.0,0.0],[1633514924149,-0.9667969,-0.020507812,0.25878906,-0.3125,-0.1875,0.21875,0.0,0.0,0.0],[1633514924179,-0.9707031,-0.015625,0.2529297,-0.125,-0.3125,0.21875,0.0,0.0,0.0],[1633514924209,-0.9765625,-0.015625,0.25390625,-0.0625,0.0,0.09375,0.0,0.0,0.0],[1633514924271,-0.9746094,-0.024414062,0.24707031,0.0,0.28125,0.40625,0.0,0.0,0.0],[1633514924299,-0.98535156,-0.018554688,0.2607422,0.03125,0.09375,0.28125,0.0,0.0,0.0],[1633514924329,-0.9863281,-0.02734375,0.25976562,0.15625,-0.125,-0.125,0.0,0.0,0.0],[1633514924392,-0.9716797,-0.020507812,0.26660156,0.09375,0.0,0.0,-51.4375,3.5,-161.125],[1633514924419,-0.9667969,-0.02734375,0.26757812,0.46875,-0.1875,0.15625,0.0,0.0,0.0],[1633514924449,-0.9814453,-0.0068359375,0.26660156,0.59375,0.0,-0.0625,0.0,0.0,0.0],[1633514924509,-0.97753906,-0.017578125,0.2607422,0.15625,-0.125,0.09375,0.0,0.0,0.0],[1633514924540,-0.9765625,-0.015625,0.2548828,-0.0625,-0.3125,0.15625,0.0,0.0,0.0],[1633514924569,-0.9785156,-0.022460938,0.25683594,0.21875,-0.25,0.0,0.0,0.0,0.0],[1633514924628,-0.9746094,-0.017578125,0.25976562,0.15625,-0.1875,-0.125,0.0,0.0,0.0],[1633514924658,-0.97558594,-0.018554688,0.265625,0.09375,0.0,-0.25,0.0,0.0,0.0],[1633514924689,-0.97753906,-0.020507812,0.26367188,0.21875,-0.375,-0.4375,0.0,0.0,0.0],[1633514924748,-0.9746094,-0.013671875,0.24902344,0.21875,-0.0625,-0.375,0.0,0.0,0.0],[1633514924779,-0.98046875,-0.020507812,0.25585938,0.09375,0.15625,-0.1875,0.0,0.0,0.0],[1633514924808,-0.9824219,-0.026367188,0.26660156,0.0,-0.3125,-0.25,0.0,0.0,0.0],[1633514924872,-0.97753906,-0.020507812,0.25683594,-0.0625,-0.25,-0.3125,-51.625,0.8125,-160.5625],[1633514924898,-0.9765625,-0.018554688,0.26953125,-0.1875,-0.25,-0.25,0.0,0.0,0.0],[1633514924928,-0.97558594,-0.025390625,0.26953125,0.0,-0.0625,-0.25,0.0,0.0,0.0],[1633514924989,-0.9765625,-0.020507812,0.25976562,0.03125,-0.375,-0.125,0.0,0.0,0.0],[1633514925020,-0.97265625,-0.025390625,0.26367188,0.34375,-0.1875,-0.3125,0.0,0.0,0.0],[1633514925049,-0.9794922,-0.021484375,0.26464844,0.21875,-0.0625,-0.125,0.0,0.0,0.0],[1633514925108,-0.9707031,-0.020507812,0.2578125,0.09375,0.0,-0.1875,0.0,0.0,0.0],[1633514925138,-0.97558594,-0.015625,0.2578125,0.09375,-0.125,-0.125,0.0,0.0,0.0],[1633514925168,-0.9814453,-0.021484375,0.2607422,0.15625,-0.125,-0.1875,0.0,0.0,0.0],[1633514925229,-0.96875,-0.01953125,0.24902344,0.40625,0.0,0.0,0.0,0.0,0.0],[1633514925261,-0.984375,-0.016601562,0.26660156,0.09375,-0.125,-0.1875,0.0,0.0,0.0],[1633514925289,-0.98046875,-0.020507812,0.26660156,-0.0625,-0.0625,-0.125,0.0,0.0,0.0],[1633514925349,-0.9873047,-0.018554688,0.25683594,-0.25,-0.125,0.0,0.0,0.0,0.0],[1633514925378,-0.98535156,-0.01953125,0.25976562,-0.125,-0.1875,-0.125,-49.9375,1.6875,-161.75],[1633514925409,-0.9746094,-0.018554688,0.25878906,0.28125,-0.25,-0.125,0.0,0.0,0.0],[1633514925469,-0.96484375,-0.021484375,0.26660156,-0.0625,-0.1875,-0.25,0.0,0.0,0.0],[1633514925498,-0.9765625,-0.0234375,0.26660156,0.09375,-0.1875,-0.25,0.0,0.0,0.0],[1633514925528,-0.9716797,-0.013671875,0.27441406,0.15625,-0.1875,-0.125,0.0,0.0,0.0],[1633514925589,-0.97558594,-0.020507812,0.26367188,0.0,0.15625,-0.0625,0.0,0.0,0.0],[1633514925620,-0.96972656,-0.018554688,0.26367188,0.09375,-0.0625,-0.125,0.0,0.0,0.0],[1633514925649,-0.9746094,-0.0234375,0.26660156,-0.125,-0.1875,0.0,0.0,0.0,0.0],[1633514925709,-0.97558594,-0.021484375,0.27148438,-0.125,-0.1875,-0.1875,0.0,0.0,0.0],[1633514925741,-0.97753906,-0.022460938,0.24707031,-0.0625,-0.25,-0.0625,0.0,0.0,0.0],[1633514925771,-0.96972656,-0.020507812,0.26171875,-0.0625,-0.25,0.0,0.0,0.0,0.0],[1633514925832,-0.9765625,-0.02734375,0.2529297,0.03125,-0.3125,0.0,0.0,0.0,0.0],[1633514925859,-0.9746094,-0.018554688,0.26367188,0.03125,-0.125,0.0,-50.5625,3.0625,-160.875],[1633514925889,-0.9746094,-0.026367188,0.25878906,0.0,-0.0625,-0.25,0.0,0.0,0.0],[1633514925949,-0.97753906,-0.030273438,0.25683594,-0.0625,-0.4375,-0.0625,0.0,0.0,0.0],[1633514925979,-0.9794922,-0.013671875,0.25585938,-0.0625,-0.25,-0.125,0.0,0.0,0.0],[1633514926009,-0.9814453,-0.018554688,0.2626953,0.09375,-0.125,0.0,0.0,0.0,0.0],[1633514926069,-0.9736328,-0.030273438,0.26367188,0.15625,-0.125,-0.0625,0.0,0.0,0.0]]'
respeck = '[[1633515913276,-0.7631836,-0.014465332,0.07977295,22.0625,9.484375,15.671875],[1633515913280,-0.79467773,0.042175293,0.15350342,7.578125,-11.6875,6.46875],[1633515913320,-0.748291,-0.07623291,0.016296387,15.78125,-9.375,4.890625],[1633515913350,-0.94970703,-0.0115356445,0.012390137,6.046875,-7.03125,1.53125],[1633515913395,-1.2099609,0.2164917,0.016296387,-19.21875,-6.078125,1.421875],[1633515913442,-1.1887207,0.03265381,0.012634277,-25.609375,-26.421875,14.359375],[1633515913470,-0.96118164,-0.2786255,-0.026184082,-21.890625,-36.015625,16.875],[1633515913516,-1.3376465,0.023864746,0.1161499,-15.59375,-4.046875,-7.640625],[1633515913564,-1.0195312,1.085144,0.113220215,-56.5,-17.421875,15.796875],[1633515913590,-1.1000977,0.15130615,-0.18341064,-44.234375,-3.75,23.234375],[1633515913650,-1.25,-0.15045166,0.36444092,-69.78125,-2.703125,-12.609375],[1633515913680,-0.9230957,0.3241577,-0.18902588,-23.125,-20.984375,-16.046875],[1633515913710,-0.8339844,0.006286621,0.010681152,-54.46875,11.3125,-3.78125],[1633515913757,-0.7492676,-0.21881104,-0.04156494,-32.9375,17.546875,1.046875],[1633515913785,-0.79418945,0.02142334,-0.13946533,-28.875,10.125,-1.640625],[1633515913830,-0.8852539,0.22869873,-0.18829346,-27.3125,-10.84375,5.3125],[1633515913875,-0.8300781,0.022644043,-0.1873169,-25.390625,0.953125,10.75],[1633515913905,-0.97558594,-0.10626221,-0.13482666,-17.53125,-9.78125,14.1875],[1633515913950,-1.2827148,0.05267334,-0.2954712,-12.734375,-10.859375,16.734375],[1633515913995,-1.3691406,-0.121398926,-0.25372314,-31.609375,-14.328125,24.6875],[1633515914025,-1.373291,0.06097412,-0.13946533,-16.296875,-4.46875,-8.28125],[1633515914072,-1.2678223,0.5787964,0.07537842,-51.078125,-21.28125,-4.953125],[1633515914115,-0.70043945,0.24505615,0.018981934,-25.734375,12.515625,7.921875],[1633515914145,-1.0493164,-0.3345337,0.27801514,-18.828125,-21.59375,-6.03125],[1633515914191,-1.1518555,0.45404053,-0.14923096,-51.65625,1.328125,-20.1875],[1633515914236,-0.83569336,0.45941162,-0.21514893,-65.78125,21.21875,10.90625],[1633515914266,-0.8544922,-0.014709473,0.19793701,-74.53125,-1.21875,13.796875],[1633515914310,-0.9790039,-0.2366333,0.042419434,-60.96875,17.78125,-3.546875],[1633515914340,-0.8847656,-0.05328369,-0.062316895,-67.625,-5.296875,-3.125],[1633515914385,-0.8898926,-0.18243408,0.057556152,-50.0,13.34375,3.9375],[1633515914430,-0.9416504,-0.11578369,0.038269043,-40.484375,-3.515625,-12.9375],[1633515914460,-0.986084,0.10003662,-0.066711426,-48.21875,-7.609375,-9.46875],[1633515914505,-1.0595703,-0.12896729,-0.024230957,-45.5,-7.359375,-2.484375],[1633515914550,-1.2456055,-0.28790283,-0.054748535,-37.890625,-24.65625,2.484375],[1633515914580,-1.317627,-0.23443604,-0.10845947,-35.046875,-29.84375,-0.28125],[1633515914625,-1.2075195,0.49578857,0.09710693,-62.609375,12.3125,-17.90625],[1633515914670,-0.85009766,0.8890991,-0.16729736,-64.578125,24.5625,-1.703125],[1633515914700,-1.1914062,-0.42095947,0.04949951,-66.484375,-51.984375,10.0],[1633515914745,-1.1728516,-0.22149658,0.14813232,-66.359375,21.078125,-49.15625],[1633515914775,-0.88500977,0.6190796,-0.24493408,-58.53125,-19.546875,-22.703125],[1633515914820,-0.8203125,-0.041809082,-0.08770752,-65.328125,16.421875,-3.796875],[1633515914865,-0.78344727,-0.26763916,-0.081848145,-54.171875,11.40625,-7.40625],[1633515914896,-0.82836914,0.14642334,-0.2041626,-42.765625,-0.609375,-10.421875],[1633515914940,-0.93408203,0.16033936,-0.21148682,-43.15625,-0.125,0.453125],[1633515914985,-0.88671875,0.0211792,-0.23858643,-42.03125,-11.0625,6.71875],[1633515915015,-0.8041992,-0.13189697,-0.22491455,-22.734375,6.28125,12.765625],[1633515915061,-1.199707,-0.25006104,-0.23736572,-17.71875,-8.140625,10.65625],[1633515915106,-1.1625977,-0.10797119,-0.2144165,-7.96875,-5.3125,18.578125],[1633515915135,-1.4638672,0.1852417,-0.1897583,-6.578125,-2.515625,-21.96875],[1633515915181,-1.5617676,0.949646,0.14202881,-55.140625,-51.890625,-19.96875]]'
json_data = {'type': 'both', 'thingy_json': thingy, 'respeck_json': respeck}

r = requests.post('http://localhost:7071/api/InferenceHttpTrigger', json=json_data)
print(r.text)