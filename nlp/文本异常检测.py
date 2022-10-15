#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import pandas as pd
import re
import os
import string
import random
import unicodedata
import numpy as np
from sklearn.ensemble import IsolationForest

USERNAME = os.getenv('USERNAME')

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
RE_IS_NUMBER = re.compile('([0-9]+)', re.U)

re_eng = re.compile('^[a-zA-Z0-9]+$', re.U)
re_not_eng = re.compile('[^a-zA-Z0-9]', re.U)
RE_ALL_ENG_SPACE = re.compile('^[a-zA-Z ,.]+$', re.U)
RE_IS_ENG = re.compile(r'([a-zA-Z]+)', re.U)
RE_IS_HAN = re.compile(r'([\u4E00-\u9FD5])', re.U)
RE_TESHUFUHAO = re.compile(r'([^\u4E00-\u9FD50-9a-zA-Z])', re.U)
# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)
RE_NOT_HAN = re.compile(r'[^\u4E00-\u9FD5]')
REGEX_HTML_RETAGS = re.compile('</?([^ >/]+).*?>', re.DOTALL | re.IGNORECASE)
NULL_SET = set(["NaN", None, "null", "nan", '', 'NULL', 'NAN'])
RE_ALL_IS_XING = re.compile(r'^\*+$')
re_is_punctuation = re.compile("[^\u4E00-\u9FD5a-zA-Z0-9]", re.U)
re_is_not_punctuation = re.compile("[\u4E00-\u9FD5a-zA-Z0-9]", re.U)

GB2312_HAN = '一丁七万丈三上下丌不与丐丑专且丕世丘丙业丛东丝丞丢两严丧丨个丫丬中丰串临丶丸丹为主丽举丿乃久乇么义之乌乍乎乏乐乒乓乔乖乘乙乜九乞也习乡书乩买乱乳乾了予争事二亍于亏云互亓五井亘亚些亟亠亡亢交亥亦产亨亩享京亭亮亲亳亵人亻亿什仁仂仃仄仅仆仇仉今介仍从仑仓仔仕他仗付仙仝仞仟仡代令以仨仪仫们仰仲仳仵件价任份仿企伉伊伍伎伏伐休众优伙会伛伞伟传伢伤伥伦伧伪伫伯估伲伴伶伸伺似伽佃但位低住佐佑体何佗佘余佚佛作佝佞佟你佣佤佥佧佩佬佯佰佳佴佶佻佼佾使侃侄侈侉例侍侏侑侔侗供依侠侣侥侦侧侨侩侪侬侮侯侵便促俄俅俊俎俏俐俑俗俘俚俜保俞俟信俣俦俨俩俪俭修俯俱俳俸俺俾倌倍倏倒倔倘候倚倜借倡倥倦倨倩倪倬倭倮债值倾偃假偈偌偎偏偕做停健偬偶偷偻偾偿傀傅傈傍傣傥傧储傩催傲傺傻像僖僚僦僧僬僭僮僳僵僻儆儇儋儒儡儿兀允元兄充兆先光克免兑兔兕兖党兜兢入全八公六兮兰共关兴兵其具典兹养兼兽冀冁冂内冈冉册再冒冕冖冗写军农冠冢冤冥冫冬冯冰冱冲决况冶冷冻冼冽净凄准凇凉凋凌减凑凛凝几凡凤凫凭凯凰凳凵凶凸凹出击凼函凿刀刁刂刃分切刈刊刍刎刑划刖列刘则刚创初删判刨利别刭刮到刳制刷券刹刺刻刽刿剀剁剂剃削剌前剐剑剔剖剜剞剡剥剧剩剪副割剽剿劁劂劈劐劓力劝办功加务劢劣动助努劫劬劭励劲劳劾势勃勇勉勋勐勒勖勘募勤勰勹勺勾勿匀包匆匈匍匏匐匕化北匙匚匝匠匡匣匦匪匮匹区医匾匿十千卅升午卉半华协卑卒卓单卖南博卜卞卟占卡卢卣卤卦卧卩卫卮卯印危即却卵卷卸卺卿厂厄厅历厉压厌厍厕厘厚厝原厢厣厥厦厨厩厮厶去县叁参又叉及友双反发叔取受变叙叛叟叠口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叻叼叽吁吃各吆合吉吊同名后吏吐向吒吓吕吖吗君吝吞吟吠吡吣否吧吨吩含听吭吮启吱吲吴吵吸吹吻吼吾呀呃呆呈告呋呐呒呓呔呕呖呗员呙呛呜呢呤呦周呱呲味呵呶呷呸呻呼命咀咂咄咆咋和咎咏咐咒咔咕咖咙咚咛咝咣咤咦咧咨咩咪咫咬咭咯咱咳咴咸咻咽咿哀品哂哄哆哇哈哉哌响哎哏哐哑哒哓哔哕哗哙哚哜哝哞哟哥哦哧哨哩哪哭哮哲哳哺哼哽哿唁唆唇唉唏唐唑唔唛唠唢唣唤唧唪唬售唯唰唱唳唷唼唾唿啁啃啄商啉啊啐啕啖啜啡啤啥啦啧啪啬啭啮啵啶啷啸啻啼啾喀喁喂喃善喇喈喉喊喋喏喑喔喘喙喜喝喟喧喱喳喵喷喹喻喽喾嗄嗅嗉嗌嗍嗑嗒嗓嗔嗖嗜嗝嗟嗡嗣嗤嗥嗦嗨嗪嗫嗬嗯嗲嗳嗵嗷嗽嗾嘀嘁嘈嘉嘌嘎嘏嘘嘛嘞嘟嘣嘤嘧嘬嘭嘱嘲嘴嘶嘹嘻嘿噌噍噎噔噗噘噙噜噢噤器噩噪噫噬噱噶噻噼嚅嚆嚎嚏嚓嚣嚯嚷嚼囊囔囗囚四囝回囟因囡团囤囫园困囱围囵囹固国图囿圃圄圆圈圉圊圜土圣在圩圪圬圭圮圯地圳圹场圻圾址坂均坊坌坍坎坏坐坑块坚坛坜坝坞坟坠坡坤坦坨坩坪坫坭坯坳坶坷坻坼垂垃垄垅垆型垌垒垓垛垠垡垢垣垤垦垧垩垫垭垮垲垴垸埂埃埋城埏埒埔埕埘埙埚埝域埠埤埭埯埴埸培基埽堀堂堆堇堋堍堑堕堙堞堠堡堤堪堰堵塄塌塍塑塔塘塞塥填塬塾墀墁境墅墉墒墓墙墚增墟墨墩墼壁壅壑壕壤士壬壮声壳壶壹夂处备复夏夔夕外夙多夜够夤夥大天太夫夭央夯失头夷夸夹夺夼奁奂奄奇奈奉奋奎奏契奔奕奖套奘奚奠奢奥女奴奶奸她好妁如妃妄妆妇妈妊妍妒妓妖妗妙妞妣妤妥妨妩妪妫妮妯妲妹妻妾姆姊始姐姑姒姓委姗姘姚姜姝姣姥姨姬姹姻姿威娃娄娅娆娇娈娉娌娑娓娘娜娟娠娣娥娩娱娲娴娶娼婀婆婉婊婕婚婢婧婪婴婵婶婷婺婿媒媚媛媪媲媳媵媸媾嫁嫂嫉嫌嫒嫔嫖嫘嫜嫠嫡嫣嫦嫩嫫嫱嬉嬖嬗嬲嬴嬷孀子孑孓孔孕字存孙孚孛孜孝孟孢季孤孥学孩孪孬孰孱孳孵孺孽宀宁它宄宅宇守安宋完宏宓宕宗官宙定宛宜宝实宠审客宣室宥宦宪宫宰害宴宵家宸容宽宾宿寂寄寅密寇富寐寒寓寝寞察寡寤寥寨寮寰寸对寺寻导寿封射将尉尊小少尔尕尖尘尚尜尝尢尤尥尧尬就尴尸尹尺尻尼尽尾尿局屁层居屈屉届屋屎屏屐屑展屙属屠屡屣履屦屮屯山屹屺屿岁岂岈岌岍岐岑岔岖岗岘岙岚岛岜岢岣岩岫岬岭岱岳岵岷岸岽岿峁峄峋峒峙峡峤峥峦峨峪峭峰峻崂崃崆崇崎崔崖崛崞崤崦崧崩崭崮崴崽崾嵇嵊嵋嵌嵘嵛嵝嵩嵫嵬嵯嵴嶂嶙嶝嶷巅巍巛川州巡巢工左巧巨巩巫差巯己已巳巴巷巽巾币市布帅帆师希帏帐帑帔帕帖帘帙帚帛帜帝带帧席帮帱帷常帻帼帽幂幄幅幌幔幕幛幞幡幢干平年并幸幺幻幼幽广庀庄庆庇床庋序庐庑库应底庖店庙庚府庞废庠庥度座庭庳庵庶康庸庹庾廉廊廑廒廓廖廛廨廪廴延廷建廾廿开弁异弃弄弈弊弋式弑弓引弗弘弛弟张弥弦弧弩弪弭弯弱弹强弼彀彐归当录彖彗彘彝彡形彤彦彩彪彬彭彰影彳彷役彻彼往征徂径待徇很徉徊律後徐徒徕得徘徙徜御徨循徭微徵德徼徽心忄必忆忉忌忍忏忐忑忒忖志忘忙忝忠忡忤忧忪快忭忮忱念忸忻忽忾忿怀态怂怃怄怅怆怊怍怎怏怒怔怕怖怙怛怜思怠怡急怦性怨怩怪怫怯怵总怼怿恁恂恃恋恍恐恒恕恙恚恝恢恣恤恧恨恩恪恫恬恭息恰恳恶恸恹恺恻恼恽恿悃悄悉悌悍悒悔悖悚悛悝悟悠患悦您悫悬悭悯悱悲悴悸悻悼情惆惊惋惑惕惘惚惜惝惟惠惦惧惨惩惫惬惭惮惯惰想惴惶惹惺愀愁愆愈愉愍愎意愕愚感愠愣愤愦愧愫愿慈慊慌慎慑慕慝慢慧慨慰慵慷憋憎憔憝憧憨憩憬憷憾懂懈懊懋懑懒懔懦懵懿戆戈戊戋戌戍戎戏成我戒戕或戗战戚戛戟戡戢戤戥截戬戮戳戴户戽戾房所扁扃扇扈扉手扌才扎扑扒打扔托扛扣扦执扩扪扫扬扭扮扯扰扳扶批扼找承技抄抉把抑抒抓投抖抗折抚抛抟抠抡抢护报抨披抬抱抵抹抻押抽抿拂拄担拆拇拈拉拊拌拍拎拐拒拓拔拖拗拘拙拚招拜拟拢拣拥拦拧拨择括拭拮拯拱拳拴拶拷拼拽拾拿持挂指挈按挎挑挖挚挛挝挞挟挠挡挢挣挤挥挨挪挫振挲挹挺挽捂捃捅捆捉捋捌捍捎捏捐捕捞损捡换捣捧捩捭据捱捶捷捺捻掀掂掇授掉掊掌掎掏掐排掖掘掠探掣接控推掩措掬掭掮掰掳掴掷掸掺掼掾揄揆揉揍揎描提插揖揞揠握揣揩揪揭揲援揶揸揽揿搀搁搂搅搋搌搏搐搓搔搛搜搞搠搡搦搪搬搭搴携搽搿摁摄摅摆摇摈摊摒摔摘摞摧摩摭摸摹摺撂撄撅撇撑撒撕撖撙撞撤撩撬播撮撰撵撷撸撺撼擀擂擅操擎擐擒擗擘擞擢擤擦攀攉攒攘攥攫攮支攴攵收攸改攻放政故效敉敌敏救敕敖教敛敝敞敢散敦敫敬数敲整敷文斋斌斐斑斓斗料斛斜斟斡斤斥斧斩斫断斯新方於施旁旃旄旅旆旋旌旎族旒旖旗无既日旦旧旨早旬旭旮旯旰旱时旷旺昀昂昃昆昊昌明昏易昔昕昙昝星映春昧昨昭是昱昴昵昶昼显晁晃晋晌晏晒晓晔晕晖晗晚晟晡晤晦晨普景晰晴晶晷智晾暂暄暇暌暑暖暗暝暧暨暮暴暹暾曙曛曜曝曦曩曰曲曳更曷曹曼曾替最月有朊朋服朐朔朕朗望朝期朦木未末本札术朱朴朵机朽杀杂权杆杈杉杌李杏材村杓杖杜杞束杠条来杨杩杪杭杯杰杲杳杵杷杼松板极构枇枉枋析枕林枘枚果枝枞枢枣枥枧枨枪枫枭枯枰枳枵架枷枸柁柃柄柏某柑柒染柔柘柙柚柜柝柞柠柢查柩柬柯柰柱柳柴柽柿栀栅标栈栉栊栋栌栎栏树栓栖栗栝校栩株栲栳样核根格栽栾桀桁桂桃桄桅框案桉桊桌桎桐桑桓桔桕桠桡桢档桤桥桦桧桨桩桫桴桶桷梁梃梅梆梏梓梗梢梦梧梨梭梯械梳梵检棂棉棋棍棒棕棘棚棠棣森棰棱棵棹棺棼椁椅椋植椎椐椒椟椠椤椭椰椴椹椽椿楂楔楗楚楝楞楠楣楦楫楮楱楷楸楹楼榀概榄榆榇榈榉榍榔榕榘榛榜榧榨榫榭榱榴榷榻槁槊槌槎槐槔槛槟槠槭槲槽槿樊樗樘樟模樨横樯樱樵樽樾橄橇橐橘橙橛橡橥橱橹橼檀檄檎檐檑檗檠檩檫檬欠次欢欣欤欧欲欷欹欺款歃歆歇歉歌歙止正此步武歧歪歹死歼殁殂殃殄殆殇殉殊残殍殒殓殖殚殛殡殪殳殴段殷殿毁毂毅毋母每毒毓比毕毖毗毙毛毡毪毫毯毳毵毹毽氅氆氇氍氏氐民氓气氕氖氘氙氚氛氟氡氢氤氦氧氨氩氪氮氯氰氲水氵永氽汀汁求汆汇汉汊汐汔汕汗汛汜汝汞江池污汤汨汩汪汰汲汴汶汹汽汾沁沂沃沅沆沈沉沌沏沐沓沔沙沛沟没沣沤沥沦沧沩沪沫沭沮沱沲河沸油治沼沽沾沿泄泅泉泊泌泐泓泔法泖泗泛泞泠泡波泣泥注泪泫泮泯泰泱泳泵泶泷泸泺泻泼泽泾洁洄洇洋洌洎洒洗洙洚洛洞津洧洪洫洮洱洲洳洵洹活洼洽派流浃浅浆浇浈浊测浍济浏浑浒浓浔浙浚浜浞浠浣浦浩浪浮浯浴海浸浼涂涅消涉涌涎涑涓涔涕涛涝涞涟涠涡涣涤润涧涨涩涪涫涮涯液涵涸涿淀淄淅淆淇淋淌淑淖淘淙淝淞淠淡淤淦淫淬淮深淳混淹添淼清渊渌渍渎渐渑渔渖渗渚渝渠渡渣渤渥温渫渭港渲渴游渺湃湄湍湎湓湔湖湘湛湟湫湮湾湿溃溅溆溉溏源溘溜溟溢溥溧溪溯溱溲溴溶溷溺溻溽滁滂滇滋滏滑滓滔滕滗滚滞滟滠满滢滤滥滦滨滩滴滹漂漆漉漏漓演漕漠漤漩漪漫漭漯漱漳漶漾潆潇潋潍潘潜潞潢潦潭潮潲潴潸潺潼澄澈澉澌澍澎澜澡澧澳澶澹激濂濉濑濒濞濠濡濮濯瀑瀚瀛瀣瀵瀹灌灏灞火灬灭灯灰灵灶灸灼灾灿炀炅炉炊炎炒炔炕炖炙炜炝炫炬炭炮炯炱炳炷炸点炻炼炽烀烁烂烃烈烊烘烙烛烟烤烦烧烨烩烫烬热烯烷烹烽焉焊焐焓焕焖焘焙焚焦焯焰焱然煅煊煌煎煜煞煤煦照煨煮煲煳煸煺煽熄熊熏熔熘熙熟熠熨熬熳熵熹燃燎燔燕燠燥燧燮燹爆爝爨爪爬爰爱爵父爷爸爹爻爽爿片版牌牍牒牖牙牛牝牟牡牢牦牧物牮牯牲牵特牺牾牿犀犁犄犊犋犍犏犒犟犬犭犯犰犴状犷犸犹狁狂狃狄狈狍狎狐狒狗狙狞狠狡狨狩独狭狮狯狰狱狲狳狴狷狸狺狻狼猁猃猊猎猓猕猖猗猛猜猝猞猡猢猥猩猪猫猬献猱猴猷猸猹猾猿獍獐獒獗獠獬獭獯獾玄率玉王玎玑玖玛玟玢玩玫玮环现玲玳玷玺玻珀珂珈珉珊珍珏珐珑珙珞珠珥珧珩班珲球琅理琉琊琏琐琚琛琢琥琦琨琪琬琮琰琳琴琵琶琼瑁瑕瑗瑙瑚瑛瑜瑞瑟瑭瑰瑶瑷瑾璀璁璃璇璋璎璐璜璞璧璨璩璺瓒瓜瓞瓠瓢瓣瓤瓦瓮瓯瓴瓶瓷瓿甄甍甏甑甓甘甙甚甜生甥用甩甫甬甭甯田由甲申电男甸町画甾畀畅畈畋界畎畏畔留畚畛畜略畦番畲畴畸畹畿疃疆疋疏疑疒疔疖疗疙疚疝疟疠疡疣疤疥疫疬疮疯疰疱疲疳疴疵疸疹疼疽疾痂痃痄病症痈痉痊痍痒痔痕痖痘痛痞痢痣痤痦痧痨痪痫痰痱痴痹痼痿瘀瘁瘃瘅瘊瘌瘐瘕瘗瘘瘙瘛瘟瘠瘢瘤瘥瘦瘩瘪瘫瘭瘰瘳瘴瘵瘸瘼瘾瘿癀癃癌癍癔癖癜癞癣癫癯癸登白百皂的皆皇皈皋皎皑皓皖皙皤皮皱皲皴皿盂盅盆盈益盍盎盏盐监盒盔盖盗盘盛盟盥目盯盱盲直相盹盼盾省眄眇眈眉看眍眙眚真眠眢眦眨眩眭眯眵眶眷眸眺眼着睁睃睇睐睑睚睛睡睢督睥睦睨睫睬睹睽睾睿瞀瞄瞅瞌瞍瞎瞑瞒瞟瞠瞢瞥瞧瞩瞪瞬瞰瞳瞵瞻瞽瞿矍矗矛矜矢矣知矧矩矫矬短矮石矶矸矽矾矿砀码砂砉砌砍砑砒研砖砗砘砚砜砝砟砣砥砦砧砩砬砭砰破砷砸砹砺砻砼砾础硅硇硌硎硐硒硕硖硗硝硪硫硬硭确硷硼碇碉碌碍碎碑碓碗碘碚碛碜碟碡碣碥碧碰碱碲碳碴碹碾磁磅磉磊磋磐磔磕磙磨磬磲磴磷磺礁礅礓礞礤礴示礻礼社祀祁祆祈祉祓祖祗祚祛祜祝神祟祠祢祥祧票祭祯祷祸祺禀禁禄禅禊福禚禧禳禹禺离禽禾秀私秃秆秉秋种科秒秕秘租秣秤秦秧秩秫秭积称秸移秽稀稂稃稆程稍税稔稗稚稞稠稣稳稷稹稻稼稽稿穆穑穗穰穴究穷穸穹空穿窀突窃窄窆窈窍窑窒窕窖窗窘窜窝窟窠窥窦窨窬窭窳窿立竖站竞竟章竣童竦竭端竹竺竽竿笃笄笆笈笊笋笏笑笔笕笙笛笞笠笤笥符笨笪笫第笮笱笳笸笺笼笾筅筇等筋筌筏筐筑筒答策筘筚筛筝筠筢筮筱筲筵筷筹筻签简箅箍箐箔箕算箜箝管箢箦箧箨箩箪箫箬箭箱箴箸篁篆篇篌篑篓篙篚篝篡篥篦篪篮篱篷篼篾簇簋簌簏簖簟簦簧簪簸簿籀籁籍米籴类籼籽粉粑粒粕粗粘粜粝粞粟粢粤粥粪粮粱粲粳粹粼粽精糁糅糇糈糊糌糍糕糖糗糙糜糟糠糨糯糸系紊素索紧紫累絮絷綦綮縻繁繇纂纛纟纠纡红纣纤纥约级纨纩纪纫纬纭纯纰纱纲纳纵纶纷纸纹纺纽纾线绀绁绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绗绘给绚绛络绝绞统绠绡绢绣绥绦继绨绩绪绫续绮绯绰绱绲绳维绵绶绷绸绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缋缌缍缎缏缑缒缓缔缕编缗缘缙缚缛缜缝缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰缱缲缳缴缵缶缸缺罂罄罅罐网罔罕罗罘罚罟罡罢罨罩罪置罱署罴罹罾羁羊羌美羔羚羝羞羟羡群羧羯羰羲羸羹羼羽羿翁翅翊翌翎翔翕翘翟翠翡翥翦翩翮翰翱翳翻翼耀老考耄者耆耋而耍耐耒耔耕耖耗耘耙耜耠耢耥耦耧耨耩耪耱耳耵耶耷耸耻耽耿聂聃聆聊聋职聍聒联聘聚聩聪聱聿肀肃肄肆肇肉肋肌肓肖肘肚肛肜肝肟肠股肢肤肥肩肪肫肭肮肯肱育肴肷肺肼肽肾肿胀胁胂胃胄胆背胍胎胖胗胙胚胛胜胝胞胡胤胥胧胨胩胪胫胬胭胯胰胱胲胳胴胶胸胺胼能脂脆脉脊脍脎脏脐脑脒脓脔脖脘脚脞脬脯脱脲脶脸脾腆腈腊腋腌腐腑腓腔腕腙腚腠腥腧腩腭腮腰腱腴腹腺腻腼腽腾腿膀膂膈膊膏膑膘膛膜膝膣膦膨膪膳膺膻臀臁臂臃臆臊臌臣臧自臬臭至致臻臼臾舀舁舂舄舅舆舌舍舐舒舔舛舜舞舟舡舢舣舨航舫般舭舯舰舱舳舴舵舶舷舸船舻舾艄艇艉艋艏艘艚艟艨艮良艰色艳艴艹艺艽艾艿节芄芈芊芋芍芎芏芑芒芗芘芙芜芝芟芡芤芥芦芨芩芪芫芬芭芮芯芰花芳芴芷芸芹芽芾苁苄苇苈苊苋苌苍苎苏苑苒苓苔苕苗苘苛苜苞苟苠苡苣苤若苦苫苯英苴苷苹苻茁茂范茄茅茆茇茈茉茌茎茏茑茔茕茗茚茛茜茧茨茫茬茭茯茱茳茴茵茶茸茹茺茼荀荃荆荇草荏荐荑荒荔荚荛荜荞荟荠荡荣荤荥荦荧荨荩荪荫荬荭荮药荷荸荻荼荽莅莆莉莎莒莓莘莛莜莞莠莨莩莪莫莰莱莲莳莴莶获莸莹莺莼莽菀菁菅菇菊菌菏菔菖菘菜菝菟菠菡菥菩菪菰菱菲菸菹菽萁萃萄萆萋萌萍萎萏萑萘萜萝萤营萦萧萨萱萸萼落葆葑著葙葚葛葜葡董葩葫葬葭葱葳葵葶葸葺蒂蒇蒈蒉蒋蒌蒎蒗蒙蒜蒡蒯蒲蒴蒸蒹蒺蒽蒿蓁蓄蓉蓊蓍蓐蓑蓓蓖蓝蓟蓠蓣蓥蓦蓬蓰蓼蓿蔌蔑蔓蔗蔚蔟蔡蔫蔬蔷蔸蔹蔺蔻蔼蔽蕃蕈蕉蕊蕖蕙蕞蕤蕨蕲蕴蕹蕺蕻蕾薄薅薇薏薛薜薤薨薪薮薯薰薷薹藁藉藏藐藓藕藜藤藩藻藿蘅蘑蘖蘧蘩蘸蘼虍虎虏虐虑虔虚虞虢虫虬虮虱虹虺虻虼虽虾虿蚀蚁蚂蚊蚋蚌蚍蚓蚕蚜蚝蚣蚤蚧蚨蚩蚪蚬蚯蚰蚱蚴蚵蚶蚺蛀蛄蛆蛇蛉蛊蛋蛎蛏蛐蛑蛔蛘蛙蛛蛞蛟蛤蛩蛭蛮蛰蛱蛲蛳蛴蛸蛹蛾蜀蜂蜃蜇蜈蜉蜊蜍蜒蜓蜕蜗蜘蜚蜜蜞蜡蜢蜣蜥蜩蜮蜱蜴蜷蜻蜾蜿蝇蝈蝉蝌蝎蝓蝗蝙蝠蝣蝤蝥蝮蝰蝴蝶蝻蝼蝽蝾螂螃螅螈螋融螓螗螟螨螫螬螭螯螳螵螺螽蟀蟆蟊蟋蟑蟒蟓蟛蟠蟥蟪蟮蟹蟾蠃蠊蠓蠕蠖蠛蠡蠢蠲蠹蠼血衄衅行衍衔街衙衡衢衣衤补表衩衫衬衮衰衲衷衽衾衿袁袂袄袅袈袋袍袒袖袜袢袤被袭袱袷袼裁裂装裆裉裎裒裔裕裘裙裟裢裣裤裥裨裰裱裳裴裸裹裼裾褂褊褐褒褓褙褚褛褡褥褪褫褰褴褶襁襄襞襟襦襻西要覃覆见观规觅视觇览觉觊觋觌觎觏觐觑角觖觚觜觞解觥触觫觯觳言訇訾詈詹誉誊誓謇謦警譬讠计订讣认讥讦讧讨让讪讫训议讯记讲讳讴讵讶讷许讹论讼讽设访诀证诂诃评诅识诈诉诊诋诌词诎诏译诒诓诔试诖诗诘诙诚诛诜话诞诟诠诡询诣诤该详诧诨诩诫诬语诮误诰诱诲诳说诵诶请诸诹诺读诼诽课诿谀谁谂调谄谅谆谇谈谊谋谌谍谎谏谐谑谒谓谔谕谖谗谘谙谚谛谜谝谟谠谡谢谣谤谥谦谧谨谩谪谫谬谭谮谯谰谱谲谳谴谵谶谷豁豆豇豉豌豕豚象豢豪豫豳豸豹豺貂貅貉貊貌貔貘贝贞负贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贳贴贵贶贷贸费贺贻贼贽贾贿赀赁赂赃资赅赆赇赈赉赊赋赌赍赎赏赐赓赔赕赖赘赙赚赛赜赝赞赠赡赢赣赤赦赧赫赭走赳赴赵赶起趁趄超越趋趑趔趟趣趱足趴趵趸趺趼趾趿跃跄跆跋跌跎跏跑跖跗跚跛距跞跟跣跤跨跪跫跬路跳践跷跸跹跺跻跽踅踉踊踌踏踔踝踞踟踢踣踩踪踬踮踯踱踵踹踺踽蹀蹁蹂蹄蹇蹈蹉蹊蹋蹑蹒蹙蹦蹩蹬蹭蹯蹰蹲蹴蹶蹼蹿躁躅躇躏躐躔躜躞身躬躯躲躺軎车轧轨轩轫转轭轮软轰轱轲轳轴轵轶轷轸轹轺轻轼载轾轿辁辂较辄辅辆辇辈辉辊辋辍辎辏辐辑输辔辕辖辗辘辙辚辛辜辞辟辣辨辩辫辰辱辶边辽达迁迂迄迅过迈迎运近迓返迕还这进远违连迟迢迤迥迦迨迩迪迫迭迮述迳迷迸迹追退送适逃逄逅逆选逊逋逍透逐逑递途逖逗通逛逝逞速造逡逢逦逭逮逯逵逶逸逻逼逾遁遂遄遇遍遏遐遑遒道遗遘遛遢遣遥遨遭遮遴遵遽避邀邂邃邈邋邑邓邕邗邙邛邝邡邢那邦邪邬邮邯邰邱邳邴邵邶邸邹邺邻邾郁郄郅郇郊郎郏郐郑郓郗郛郜郝郡郢郦郧部郫郭郯郴郸都郾鄂鄄鄙鄞鄢鄣鄯鄱鄹酃酆酉酊酋酌配酎酏酐酒酗酚酝酞酡酢酣酤酥酩酪酬酮酯酰酱酲酴酵酶酷酸酹酽酾酿醅醇醉醋醌醍醐醑醒醚醛醢醣醪醭醮醯醴醵醺采釉释里重野量金釜鉴銎銮鋈錾鍪鎏鏊鏖鐾鑫钅钆钇针钉钊钋钌钍钎钏钐钒钓钔钕钗钙钚钛钜钝钞钟钠钡钢钣钤钥钦钧钨钩钪钫钬钭钮钯钰钱钲钳钴钵钶钷钸钹钺钻钼钽钾钿铀铁铂铃铄铅铆铈铉铊铋铌铍铎铐铑铒铕铖铗铘铙铛铜铝铞铟铠铡铢铣铤铥铧铨铩铪铫铬铭铮铯铰铱铲铳铴铵银铷铸铹铺铼铽链铿销锁锂锃锄锅锆锇锈锉锊锋锌锍锎锏锐锑锒锓锔锕锖锗锘错锚锛锝锞锟锡锢锣锤锥锦锨锩锪锫锬锭键锯锰锱锲锴锵锶锷锸锹锺锻锼锾锿镀镁镂镄镅镆镇镉镊镌镍镎镏镐镑镒镓镔镖镗镘镙镛镜镝镞镟镡镢镣镤镥镦镧镨镩镪镫镬镭镯镰镱镲镳镶长门闩闪闫闭问闯闰闱闲闳间闵闶闷闸闹闺闻闼闽闾阀阁阂阃阄阅阆阈阉阊阋阌阍阎阏阐阑阒阔阕阖阗阙阚阜阝队阡阢阪阮阱防阳阴阵阶阻阼阽阿陀陂附际陆陇陈陉陋陌降限陔陕陛陟陡院除陧陨险陪陬陲陴陵陶陷隅隆隈隋隍随隐隔隗隘隙障隧隰隳隶隹隼隽难雀雁雄雅集雇雉雌雍雎雏雒雕雠雨雩雪雯雳零雷雹雾需霁霄霆震霈霉霍霎霏霓霖霜霞霪霭霰露霸霹霾青靓靖静靛非靠靡面靥革靳靴靶靼鞅鞋鞍鞑鞒鞔鞘鞠鞣鞫鞭鞯鞲鞴韦韧韩韪韫韬韭音韵韶页顶顷顸项顺须顼顽顾顿颀颁颂颃预颅领颇颈颉颊颌颍颏颐频颓颔颖颗题颚颛颜额颞颟颠颡颢颤颥颦颧风飑飒飓飕飘飙飚飞食飧飨餍餐餮饔饕饣饥饧饨饩饪饫饬饭饮饯饰饱饲饴饵饶饷饺饼饽饿馀馁馄馅馆馇馈馊馋馍馏馐馑馒馓馔馕首馗馘香馥馨马驭驮驯驰驱驳驴驵驶驷驸驹驺驻驼驽驾驿骀骁骂骄骅骆骇骈骊骋验骏骐骑骒骓骖骗骘骚骛骜骝骞骟骠骡骢骣骤骥骧骨骰骱骶骷骸骺骼髀髁髂髅髋髌髑髓高髟髡髦髫髭髯髹髻鬃鬈鬏鬓鬟鬣鬯鬲鬻鬼魁魂魃魄魅魇魈魉魍魏魑魔鱼鱿鲁鲂鲅鲆鲇鲈鲋鲍鲎鲐鲑鲒鲔鲕鲚鲛鲜鲞鲟鲠鲡鲢鲣鲤鲥鲦鲧鲨鲩鲫鲭鲮鲰鲱鲲鲳鲴鲵鲶鲷鲸鲺鲻鲼鲽鳃鳄鳅鳆鳇鳊鳋鳌鳍鳎鳏鳐鳓鳔鳕鳖鳗鳘鳙鳜鳝鳞鳟鳢鸟鸠鸡鸢鸣鸥鸦鸨鸩鸪鸫鸬鸭鸯鸱鸲鸳鸵鸶鸷鸸鸹鸺鸽鸾鸿鹁鹂鹃鹄鹅鹆鹇鹈鹉鹊鹋鹌鹎鹏鹑鹕鹗鹘鹚鹛鹜鹞鹣鹤鹦鹧鹨鹩鹪鹫鹬鹭鹰鹱鹳鹾鹿麂麇麈麋麒麓麝麟麦麴麸麻麽麾黄黉黍黎黏黑黔默黛黜黝黟黠黢黥黧黩黪黯黹黻黼黾鼋鼍鼎鼐鼓鼗鼙鼠鼢鼬鼯鼷鼹鼻鼽鼾齄齐齑齿龀龃龄龅龆龇龈龉龊龋龌龙龚龛龟龠'

unicodedata.normalize('NFKD', '，')  # 将文本标准化

VEHICLE_NO_RE_COMPILE = re.compile(
    '^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}[A-Z]{1}(([A-HJ-NP-Z0-9]{5}[DF]{1})|([DF]{1}[A-HJ-NP-Z0-9]{5})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})$')

HTML_SET = {'<h>', '<sup>', '<cite>', '<menuitem>', '<mark>', '<noscript>', '<figure>', '<colgroup>', '<style>', '<abbr>', '<progress>', '<dfn>', '<applet>', '<bdo>', '<a>', '<form>', '<area>', '<strike>', '<track>', '<tt>', '<img>', '<article>', '<embed>', '<iframe>', '<b>', '<canvas>', '<ol>', '<nav>', '<thead>', '<acronym>', '<s>', '<bdi>', '<dd>', '<label>', '<center>', '<datalist>', '<p>', '<audio>', '<option>', '<address>', '<tfoot>', '<table>', '<blockquote>', '<th>', '<kbd>', '<rp>', '<param>', '<video>', '<details>', '<dl>', '<section>', '<!-->', '<frameset>', '<ruby>', '<title>', '<figcaption>', '<meter>', '<sub>', '<dialog>', '<time>', '<del>', '<samp>', '<code>', '<pre>', '<wbr>', '<col>', '<map>', '<tr>', '<li>', '<source>', '<select>', '<strong>', '<body>', '<meta>', '<noframes>', '<main>', '<div>', '<object>', '<tbody>', '<footer>', '<h1> - <h6>', '<dt>', '<html>', '<rt>', '<i>', '<head>', '<header>', '<br>', '<legend>', '<caption>', '<td>', '<span>', '<menu>', '<hr>', '<small>', '<big>', '<command>', '<ul>', '<summary>', '<optgroup>', '<link>', '<dir>', '<aside>', '<em>', '<input>', '<basefont>', '<script>', '<q>', '<fieldset>', '<base>', '<var>', '<u>', '<frame>', '<button>', '<!DOCTYPE>', '<textarea>', '<font>', '<ins>', '<keygen>', '<output>'}
HTML_SET = set(t[1:-1] for t in HTML_SET)
RE_HTML_SET = re.compile(r'\<(({}))'.format(')|('.join(list(HTML_SET))))

GB2312_HAN_SET = set(GB2312_HAN+string.digits+string.ascii_letters)

MAX_SEQ_LEN = 128  # 最大处理字符串长度，若超过此长度，则截断；


def is_vehicle_no(text):
    """
    车牌号的识别
    :param text:
    :return:
    """
    return re.search(VEHICLE_NO_RE_COMPILE, text)


def remove_tags(text, which_ones=(), keep=(), encoding=None):
    """

    :param text:
    :param which_ones:
    :param keep:
    :param encoding:
    :return:
    """
    if which_ones and keep:
        raise ValueError('Cannot use both which_ones and keep')

    if '<' not in text or not re.search(RE_HTML_SET, text):
        # 没有html标签时候不去删除，防止将 “学习<高等数学>”类数据删除了；
        return text

    which_ones = {tag.lower() for tag in which_ones}
    keep = {tag.lower() for tag in keep}

    def will_remove(tag):
        tag = tag.lower()
        if which_ones:
            return tag in which_ones
        else:
            return tag not in keep

    def remove_tag(m):
        tag = m.group(1)
        return u'' if will_remove(tag) else m.group(0)

    return REGEX_HTML_RETAGS.sub(remove_tag, to_unicode(text, encoding))


def to_unicode(text, encoding=None, errors='strict'):
    """Return the unicode representation of a bytes object `text`. If `text`
    is already an unicode object, return it as-is."""
    if isinstance(text, str):
        return text
    if not isinstance(text, (bytes, str)):
        raise TypeError('to_unicode must receive a bytes, str or unicode '
                        'object, got %s' % type(text).__name__)
    if encoding is None:
        encoding = 'utf-8'
    return text.decode(encoding, errors)


def is_number(s):
    try:
        if isinstance(s, str):
            # 剔除掉符号；
            s = re.sub(re_is_punctuation, '', s)
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


CHANGDUYICHANG_MSG = "长度异常"
CHONGFUZIYICHANG_MSG = "重复字异常"
SHUZIYICHANG_MSG = "数字异常"
DIPINZIYICHANG_MSG = '低频字异常'
GAOPINZIYICHANG_MSG = "高频字异常"
TESHUZIFUYICHANG_MSG = '特殊字符异常'
ZIMUYICHANG_MSG = "字母异常"
HANZIYICHANG_MSG = "汉字异常"
FENCIYICHANG_MSG = "分词异常"
ZIPINYICHANG_MSG = "字频异常"
XUNLIANYANGBENTAISHAO_MSG = "训练样本太少"
XUNLIANSHIFEIWENBENGUODUO_MSG = "训练时非文本过多"
XUNLIANSHIWENBENNEIRONGGUOSHAO_MSG = "训练时文本内容过少"
DUOTEZHENGRONGHEYICHANG_MSG = "多特征融合异常"
MOXINGPANDINGYICHANG_MSG = "模型判定异常"
FEIWENBENLEIXINGSHUJU_MSG = "非文本类型数据"
SHURUWEIQUESHIZHI_MSG = "输入为缺失值"
LUANMALVGAO_MSG = "乱码率高"
HTMLGUODUO_MSG = "HTML过多"
TESHUFUHAOGUODUO_MSG = "特殊符号过多"
MOXINGWEIXUNLIAN_MSG = "模型未训练"
SHURUSHUJULEIXINGCUOWU_MSG = "输入数据类型错误"
WENBENSHUJUYICHANG_MSG = "文本数据异常"

ANOMALY_TO_PINYIN = {
        CHANGDUYICHANG_MSG: "chang_du_yi_chang",
        CHONGFUZIYICHANG_MSG: "chong_fu_zi_yi_chang",
        SHUZIYICHANG_MSG: "shu_zi_yi_chang",
        DIPINZIYICHANG_MSG: "di_pin_zi_yi_chang",
        GAOPINZIYICHANG_MSG: "gao_pin_zi_yi_chang",
        TESHUZIFUYICHANG_MSG: "te_shu_zi_fu_yi_chang",
        ZIMUYICHANG_MSG: "zi_mu_yi_chang",
        HANZIYICHANG_MSG: "han_zi_yi_chang",
        FENCIYICHANG_MSG: "fen_ci_yi_chang",
        ZIPINYICHANG_MSG: "zi_pin_yi_chang",
        XUNLIANYANGBENTAISHAO_MSG: "xun_lian_yang_ben_tai_shao",
        XUNLIANSHIFEIWENBENGUODUO_MSG: "xun_lian_shi_fei_wen_ben_guo_duo",
        XUNLIANSHIWENBENNEIRONGGUOSHAO_MSG: "xun_lian_shi_wen_ben_nei_rong_guo_shao",
        DUOTEZHENGRONGHEYICHANG_MSG: "duo_te_zheng_rong_he_yi_chang",
        MOXINGPANDINGYICHANG_MSG: "mo_xing_pan_ding_yi_chang",
        FEIWENBENLEIXINGSHUJU_MSG: "fei_wen_ben_lei_xing_shu_ju",
        SHURUWEIQUESHIZHI_MSG: "shu_ru_wei_que_shi_zhi",
        LUANMALVGAO_MSG: "luan_ma_lv_gao",
        HTMLGUODUO_MSG: "HTML_guo_duo",
        TESHUFUHAOGUODUO_MSG: "te_shu_fu_hao_guo_duo",
        MOXINGWEIXUNLIAN_MSG: "mo_xing_wei_xun_lian",
        SHURUSHUJULEIXINGCUOWU_MSG: "shu_ru_shu_ju_lei_xing_cuo_wu",
        WENBENSHUJUYICHANG_MSG: "wen_ben_shu_ju_yi_chang",
}

ANOMALY_EXPLAIN_FIELDS = [
                          CHANGDUYICHANG_MSG,
                          # CHONGFUZIYICHANG_MSG,
                          SHUZIYICHANG_MSG,
                          # DIPINZIYICHANG_MSG,
                          GAOPINZIYICHANG_MSG,
                          TESHUZIFUYICHANG_MSG,
                          ZIMUYICHANG_MSG,
                          HANZIYICHANG_MSG,
                          # FENCIYICHANG_MSG,
                          # ZIPINYICHANG_MSG
                          ]


def get_text_features(t, p25_words, p75_words, words_count):
    """
    生成文本的特征
    :param t:
    :return:
    """
    t = "{}".format(t)
    # 总长，去重总长，数字，低频词，高频词, 标点符号数，字母数，汉字数，分词数

    if re.search(RE_ALL_ENG_SPACE, t):
        # 全是英文单词及空格
        len_t = len(t.split())
        explain_features = [
            len_t,
            # len(set(t.split())),
            len(re.findall(RE_IS_NUMBER, t)) / len_t,
            # len([w for w in t if w in p25_words]),
            len([w for w in t if w in p75_words]),
            0,
            len(re.findall(RE_IS_ENG, t)) / len_t,
            len(re.findall(RE_IS_HAN, t)) / len_t,
            # len(t.split()),
            # sum([words_count.get(w, 0) for w in t]),
        ]
    else:
        len_t = len(t)
        explain_features = [
            len_t,
            # len(set(t)),
            len(re.findall(RE_IS_NUMBER, t)) / len_t,
            # len([w for w in t if w in p25_words]),
            len([w for w in t if w in p75_words]),
            len(re.findall(RE_TESHUFUHAO, t)) / len_t,
            len(re.findall(RE_IS_ENG, t)) / len_t,
            len(re.findall(RE_IS_HAN, t)) / len_t,
            # len(jieba.lcut(t)),
            # sum([words_count.get(w, 0) for w in t]),
        ]

    return explain_features


def get_percentile_contamination(datas):
    """
    根据分位数，用户设置样本中异常点的比例
    :param datas:
    :return:
    """
    max_ = max(datas)
    min_ = min(datas)
    if max_ != min_:
        datas = [int(10 * (t - min_) / (max_ - min_)) for t in datas]

    percentile = np.percentile(datas, (1, 5, 10, 25, 50, 75, 90, 95, 99), interpolation='nearest')
    # interpolation 可选参数指定了当需要的百分比位于两个数据点i < j
    # 之间时使用的插值方法:
    # 1）‘linear’: i + (j - i) * fraction,  fraction是由i和j包围的索引的分数部分。
    # 2）‘lower’: i.
    # 3）‘higher’: j.
    # 4）‘nearest’: i或j，以最接近的为准。
    # 5）‘midpoint’: (i + j) / 2.

    # if len([t for t in datas if t < percentile[0] or t > percentile[-1]]) > 100:
    #     contamination = 0.005
    # el
    if percentile[1] != percentile[2] and percentile[-2] != percentile[-3]:
        contamination = 0.1
    elif percentile[1] == percentile[2] and percentile[-2] != percentile[-3]:
        contamination = 0.05
    elif percentile[1] != percentile[2] and percentile[-2] == percentile[-3]:
        contamination = 0.05
    elif percentile[0] != percentile[1] and percentile[-1] != percentile[-2]:
        contamination = 0.02
    elif percentile[0] == percentile[1] and percentile[-1] != percentile[-2]:
        contamination = 0.01
    elif percentile[0] != percentile[1] and percentile[-1] == percentile[-2]:
        contamination = 0.01
    else:
        contamination = 0.005

    return contamination

def is_enum_data(num_unique, data_size, df_all_features=None):
    """
    判断数据集是否是枚举类数据集；
    :param num_unique: 数据集唯一值的个数；
    :param data_size: 数据集大小；
    :param df_all_features: 数据集的特征值
    :return:
    """
    if num_unique/data_size < 0.01 or (num_unique/data_size < 0.1 and num_unique < 1000):
        # 唯一数据比例少, 则认为是枚举类型数据；
        return True
    else:
        return False

LEN_INDEX = ANOMALY_EXPLAIN_FIELDS.index(CHANGDUYICHANG_MSG)
SHUZI_INDEX = ANOMALY_EXPLAIN_FIELDS.index(SHUZIYICHANG_MSG)
ZIMU_INDEX = ANOMALY_EXPLAIN_FIELDS.index(ZIMUYICHANG_MSG)
FUHAO_INDEX = ANOMALY_EXPLAIN_FIELDS.index(TESHUZIFUYICHANG_MSG)
GAOPINZI_INDEX = ANOMALY_EXPLAIN_FIELDS.index(GAOPINZIYICHANG_MSG)
HANZI_INDEX = ANOMALY_EXPLAIN_FIELDS.index(HANZIYICHANG_MSG)

def enum_anomaly(unique_datas, df_all_features, mean_std_list=None):
    """
    枚举类数据的异常检测；
    针对枚举类数据，不能简单的通过高频、低频等预测；如所在城市，有时候有的城市出现次数少，但也是存在的，故不能认为异常；
    故针对于此，对枚举类数据，检测上需特殊处理；

    :param unique_datas:
    :param df_all_features:
    :return: 返回一个枚举类型的检测函数；
    """
    if mean_std_list is None:# 枚举类数据异常，仅仅是根据最大值，最小值，均值，标准差判断；
        mean_std_list = [[mean, std] for mean, std in zip(df_all_features.mean(), df_all_features.std())]
        # print('mean_std_list={}'.format(mean_std_list))
    if mean_std_list[SHUZI_INDEX][1] < 0.01 or mean_std_list[ZIMU_INDEX][1] < 0.01 or mean_std_list[FUHAO_INDEX][1] < 0.01:
        # 数字，字母，符号很少，仅仅根据文本长度来判断异常；
        mean, std = mean_std_list[LEN_INDEX]
        return lambda x: False if mean - 20 * std < len(x) < mean + 20 * std else True
    else:
        return lambda x: x in unique_datas

def features_random_noise_injection(df_all_features, random_noise_injection):
    """
    对特征数据注入噪音
    :param df_all_features:
    :param random_noise_injection: 注入噪音的比例
    :return:
    """
    random_num = int(df_all_features.shape[0] * random_noise_injection)

    if random_num <= 10:
        return df_all_features

    if not isinstance(df_all_features, pd.DataFrame):
        df_all_features = pd.DataFrame(df_all_features)

    df_all_features_mean_std = [[int(( - 2 * std)*1000), int(( + 2 * std)*1000)] for mean, std in
                                zip(df_all_features.mean(axis=0), df_all_features.std(axis=0))]

    row_choice = random.sample([i for i in range(df_all_features.shape[0])], random_num)
    col_choice = [i for i in range(df_all_features.shape[1])]
    for row in row_choice:
        col = random.choice(col_choice)
        min_, max_ = df_all_features_mean_std[col]
        v = random.randint(min_, max_)/1000
        if random.random() > 0.5:
            df_all_features.iloc[row, col] += v
        else:
            df_all_features.iloc[row, col] -= v

    df_all_features[df_all_features < 0] = 0

    return df_all_features

class ModelTextOutlierDetectionClass(object):
    """
    文本异常检测；
    1，对输入的文本样本进行训练，若样本太少，非文本样本过多，会提示出错；
    2，依据训练的结果，对输入的样本进行异常预测，少数情况下，不需要训练的结果就可以判断异常与否，大多数情况下需要依赖训练的结果；
    即，若在预测的之前，没有进行对应的训练，则预测出错；
    """

    def __init__(self, choice_contamination=None, random_model_number=1, min_n_estimators=10, max_n_estimators=150):
        """

        :param choice_contamination: 可选择的取样比例列表；
        :param random_model_number: 随机生成的模型数
        :param min_n_estimators: 最小随机树数目；
        :param max_n_estimators: 最大随机树数目；
        """
        if choice_contamination is None:
            self.choice_contamination = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 2e-4,
                                         1e-4, 5e-5, 1e-5, 1e-6]
        else:
            self.choice_contamination = choice_contamination
        self.random_model_number = random_model_number
        self.min_n_estimators = min_n_estimators
        self.max_n_estimators = max_n_estimators
        self.normed = False
        self.one_feature_model = False  # 若为真，则每个字段，单独训练一个模型；
        self.initialization()

    def initialization(self):
        self.iforest = None
        self.samples_iforest = []
        self.model_error = ''
        self.p25_words = ''
        self.p75_words = ''
        self.words_count = {}
        self.all_features_min = None
        self.all_features_max = None
        self.enum_func = None  # 枚举类数据异常判断函数；
        self.punctuation_too_many = False  # 若训练集中特殊符号太多，则预测时候不检测特殊符号及乱码率；
        self.not_all_feature = False  # 针对特殊数据集，个别特征，如长度特征、高频字特征等；
        self.use_features_columns = [ANOMALY_EXPLAIN_FIELDS.index(col) for col in ANOMALY_EXPLAIN_FIELDS] # 使用哪些特征列；针对特殊数据集，不检测长度特征；有些数据集高频字为空，这时不检查高频字特征；针对长文本内容，不验证高频字特征；
        self.calibration_value = 1e-5  # 对阈值进行校正；
        self.random_noise_injection = 0.0  # 随机噪音注入； # 向样本中注入噪音以达到增强模型的鲁棒性的效果

    def text_standard(self, text):
        # 全英文字符时候，不删除空格；
        text = unicodedata.normalize('NFKD', text).strip()[:MAX_SEQ_LEN]
        if re.search(RE_ALL_ENG_SPACE, text):
            return text
        else:
            return text.replace(' ', '')

    def fit(self, X, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        # time1 = time.time()
        self.initialization()
        if len(X) < 5:
            self.model_error = XUNLIANYANGBENTAISHAO_MSG
            return False

        digits_num = len([text for text in X if is_number(text)])
        # time22 = time.time()
        # print('数值检查完成：{}'.format(time22-time1))

        if digits_num > len(X) * 0.8:
            self.model_error = XUNLIANSHIFEIWENBENGUODUO_MSG
            return False

        datas = [self.text_standard(text) for text in X if isinstance(text, str)]
        datas = [text for text in datas if text and text.strip() and not any(
            k == text.strip() for k in ['不详', '未知', '空', '-', '\\N', '未填写', 'NULL', 'null', 'NaN', 'nan']) and not re.search(RE_ALL_IS_XING, text)]

        if len(datas) < len(X) * 0.2:
            self.model_error = XUNLIANSHIWENBENNEIRONGGUOSHAO_MSG
            return False

        # time2 = time.time()
        # print('空值检查完成：{}'.format(time2-time22))

        words_count = {}

        for text in datas:
            for word in text:
                words_count.setdefault(word, 0)
                words_count[word] += 1

        # time3 = time.time()
        # print('词频统计完成：{}'.format(time3-time2))
        # words_count = collections.Counter(''.join(address))

        # 10%, 90分位数
        # p25, p75 = np.percentile(list(words_count.values()), [20, 80])
        # self.p25_words = set(w for w, v in words_count.items() if v <= p25)
        # self.p75_words = set(w for w, v in words_count.items() if v >= p75)
        data_size = len(datas)
        self.p25_words = set([])
        self.p75_words = set(w for w, v in words_count.items() if v >= data_size * 0.8)

        if not self.p75_words:
            # 高频字为空时候，不检查所有特征；
            self.not_all_feature = True
            self.use_features_columns.remove(GAOPINZI_INDEX)

        # 分档计分；, np.percentile(datas, (1, 5, 10, 25, 50, 75, 90, 95, 99), interpolation='nearest')
        # max_words_count = max(words_count.values())
        # min_words_count = min(words_count.values())
        # if max_words_count == min_words_count:
        #     max_words_count = min_words_count +100
        # self.words_count = {k:(v-min_words_count)/(max_words_count-min_words_count) for k, v in words_count.items()}
        self.words_count = words_count

        punctuation_data_size = sum([len(re.findall(re_is_punctuation, text)) for text in datas])
        if punctuation_data_size/(data_size+1) > 1.2 and data_size > 1e4:
            # 几乎每个数据都有标点符号，且数据总量过万；这个时候预测的时候，不检测乱码及特殊符号
            self.punctuation_too_many = True

        # time4 = time.time()
        # print('分位数计算完成：{}'.format(time4-time3))

        anomaly_explain = ANOMALY_EXPLAIN_FIELDS

        all_features = [get_text_features(text, self.p25_words, self.p75_words, self.words_count) for text in datas]

        assert len(anomaly_explain) == len(all_features[0]), '【错误】特征与特征说明不一致！'

        # anomaly_explain, all_features = get_text_features(datas, self.p25_words, self.p75_words, self.words_count)
        df_all_features = pd.DataFrame(all_features)

        unique_datas = set(datas)
        if is_enum_data(len(unique_datas), data_size):

            mean_std_list = [[mean, std] for mean, std in zip(df_all_features.mean(), df_all_features.std())]
            if mean_std_list[SHUZI_INDEX][1] >= 0.01 or mean_std_list[ZIMU_INDEX][1] >= 0.01 or mean_std_list[FUHAO_INDEX][
                1] >= 0.01 or mean_std_list[GAOPINZI_INDEX][1] > 0.01:
                # 枚举类型，数字、字母、符号、高频字特征有一个显著的时候，就去除长度特征；
                self.not_all_feature = True
                self.use_features_columns.remove(LEN_INDEX)
                del unique_datas
                self.enum_func = None
            else:
                # 枚举类型，数字、字母、符号特征有一个不显著的时候，就仅仅检测长度异常；
                enum_func = enum_anomaly(unique_datas, df_all_features, mean_std_list=mean_std_list)
                self.enum_func = enum_func
                return self.enum_func
        elif df_all_features.mean()[0] > 50:
            # 字符串平均长度大于15时候，统计高频字、字母、数字特征都会出现不太准确的情况；比如公司简介、微博评论，这个时候检查高频字个数较少，若碰到较短语句且高频字较多，可能就判别为异常；
            # 这个时候直接调整对应的阈值
            self.calibration_value = 0.02
            self.random_noise_injection = 0.1
        elif df_all_features.mean()[0] > 15:
            # 字符串平均长度大于15时候，统计高频字、字母、数字特征都会出现不太准确的情况；比如公司简介、微博评论，这个时候检查高频字个数较少，若碰到较短语句且高频字较多，可能就判别为异常；
            # 这个时候直接调整对应的阈值
            self.calibration_value = 0.01
            self.random_noise_injection = 0.05
        elif df_all_features.mean()[0] > 7:
            # 字符串平均长度大于15时候，统计高频字、字母、数字特征都会出现不太准确的情况；比如公司简介、微博评论，这个时候检查高频字个数较少，若碰到较短语句且高频字较多，可能就判别为异常；
            # 这个时候直接调整对应的阈值
            self.calibration_value = 0.005
            self.random_noise_injection = 0.01
        else:
            del unique_datas
            self.enum_func = None

        # time5 = time.time()
        # print('特征值提取完成：{}'.format(time5-time4))

        if self.normed:
            # 归一化
            np_all_features = np.array(all_features)
            self.all_features_min = np_all_features.min(axis=0)  # 列最小值
            self.all_features_max = np_all_features.max(axis=0)  # 列最大值

        # time6 = time.time()
        # print('归一化完成：{}'.format(time6-time5))

        self.samples_iforest = []
        # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        if self.one_feature_model:
            # 不随机生成参数模型；每列特征值取一个模型；
            df_all_features = pd.DataFrame(all_features)
            for col_index, anomaly_name in enumerate(anomaly_explain):
                # print('开始训练模型：{}'.format(anomaly_name))
                # time7 = time.time()
                contamination = get_percentile_contamination(df_all_features[col_index].values)
                contamination /= 8

                # print(contamination, len(all_features))

                if len(all_features) > 2e5:
                    # 样本数超过20万的时候，随机选择5万；
                    sample_features = random.sample(all_features, int(2e5))
                else:
                    sample_features = all_features
                sample_features = pd.DataFrame(sample_features)

                while (sample_features.shape[0] * contamination > 1000):
                    contamination /= 10

                sample_features_normed = [t for t in sample_features[[col_index, 0]].values]

                if len(sample_features_normed) > 256:
                    max_samples = max(256, int(5 / contamination))  # 根据总样本数，设置每棵树样本个数或比例
                else:
                    max_samples = len(sample_features_normed)

                iforest = IsolationForest(max_samples=max_samples, contamination=contamination, bootstrap=True,
                                          n_jobs=1)
                # bootstrap,True,有放回采样；
                # n_jobs,处理器数据，默认是1;设置为-1则表示使用所有可用的处理器,在本机测试，设置为-1，训练速度反而翻倍；
                iforest.fit(sample_features_normed)
                if hasattr(iforest, 'offset_'):
                    iforest.offset_ = iforest.offset_ - self.calibration_value
                elif hasattr(iforest, 'threshold_'):
                    iforest.threshold_ = iforest.threshold_ - self.calibration_value
                self.samples_iforest.append([anomaly_name, iforest])
                # print('训练模型{}，耗时：{}'.format(anomaly_name, time.time()-time7))
        else:
            # 随机生成系列参数，并生成随机模型；
            for anomaly_name in range(self.random_model_number):
                # print('开始训练模型：{}'.format(anomaly_name))
                # time7 = time.time()

                # n_estimators = random.randint(self.min_n_estimators, self.max_n_estimators)
                n_estimators = 100
                # choice_contamination = [contamination for contamination in self.choice_contamination if len(datas) * contamination > 1]
                # contamination = random.choice(choice_contamination)
                contamination = 0.005
                # print(n_estimators, contamination, len(all_features))

                if len(all_features) > 2e5:
                    # 样本数超过20万的时候，随机选择20万；
                    sample_features = random.sample(all_features, int(2e5))
                else:
                    sample_features = all_features
                sample_features = np.array(sample_features)
                if self.normed:
                    # 归一化
                    sample_features_normed_col = []
                    for index, col in enumerate(sample_features.T):
                        max_col, min_col = self.all_features_max[index], self.all_features_min[index]
                        if (max_col - min_col) == 0:
                            sample_features_normed_col.append([1 for _ in col])
                        else:
                            sample_features_normed_col.append([(c - min_col) / (max_col - min_col) for c in col])
                    sample_features_normed_col = np.array(sample_features_normed_col)
                    sample_features_normed = sample_features_normed_col.T
                else:
                    sample_features_normed = sample_features
                max_features = sample_features_normed.shape[1]
                # 指定从总样本X中抽取来训练每棵树iTree的属性的数量，默认只使用一个属性
                while (sample_features_normed.shape[0] * contamination >= 1000):
                    contamination /= 3

                if sample_features_normed.shape[0] < 100 and contamination <= 0.01:
                    contamination = 0.1

                if len(sample_features_normed) > 256:
                    max_samples = max(256, int(5 / contamination))  # 根据总样本数，设置每棵树样本个数或比例
                else:
                    max_samples = len(sample_features_normed)

                if self.not_all_feature:
                    # 去除长度特征
                    sample_features_normed = sample_features_normed[:, self.use_features_columns]
                    max_features = len(self.use_features_columns)

                # 注入噪音
                sample_features_normed = features_random_noise_injection(sample_features_normed, self.random_noise_injection)

                iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                          max_features=max_features, max_samples=max_samples)
                iforest.fit(sample_features_normed)
                if hasattr(iforest, 'offset_'):
                    iforest.offset_ = iforest.offset_ - self.calibration_value
                elif hasattr(iforest, 'threshold_'):
                    iforest.threshold_ = iforest.threshold_ - self.calibration_value

                self.samples_iforest.append([DUOTEZHENGRONGHEYICHANG_MSG, iforest])
                # print('训练模型{}，耗时：{}'.format(anomaly_name, time.time() - time7))
        self.iforest = self.samples_iforest
        return self.iforest

    def samples_predict(self, X, decision=False):
        """
        对文本进行预测，返回异常原因及是否异常（1，正常，-1，异常）；
        若decision真，则返回具体分数[[得分,阈值]]；
        :param X:
        :param decision:
        :return:
        """
        all_is_inlier = []
        all_threshold_ = []
        all_decision_scores = []
        anomaly_model_list = []
        data_size = len(X)
        for col_index, (anomaly_name, iforest) in enumerate(self.samples_iforest):
            anomaly_model_list.append(anomaly_name)
            if self.one_feature_model:
                df_x = pd.DataFrame(X)
                test_x = [t for t in df_x[[col_index, 0]].values]
            elif self.not_all_feature:
                # 去除长度特征
                np_X = np.array(X)
                test_x = np_X[:, self.use_features_columns]
            else:
                test_x = X

            if decision:
                scores = iforest.decision_function(test_x)   # 该函数计算的得分区间为[-0.5, 0.5]
                all_threshold_.append(iforest.threshold_)
                all_decision_scores.append(scores)
            else:
                is_inlier = iforest.predict(test_x)
                all_is_inlier.append(is_inlier)
        if decision:
            # 每个模型的值加和，再与整体均值比较；
            all_decision_scores = np.array(all_decision_scores)
            mean_threshold = sum(all_threshold_)
            # threshold_ = - self.random_model_number*0.6  # 有80% 判断为-1，则认为-1
            # pred_list = [-1 if sum(all_decision_scores[:, i]) <= mean_threshold else 1 for i in range(data_size)]
            # return [[MOXINGPANDINGYICHANG_MSG, pred] for pred in pred_list]
            return [[sum(all_decision_scores[:, i]), mean_threshold] for i in range(data_size)]
        else:
            # 根据每个模型的判定，再取多数；
            all_is_inlier = np.array(all_is_inlier)
            if self.one_feature_model:
                # threshold_ = - len(self.samples_iforest)*0  # 单字段预测时，有50% 判断为-1，则认为-1
                # threshold_ = len(self.samples_iforest) - 2 # 有2个模型判定异常就认定是异常；
                # pred_list = [-1 if sum(all_is_inlier[:, i]) <= threshold_ else 1 for i in range(len(X))]
                explain_pred_list = []
                for i in range(data_size):
                    error_preds = [e_i for e_i, e_v in enumerate(all_is_inlier[:, i]) if e_v == -1]
                    if error_preds:
                        explain_pred_list.append(['|'.join([anomaly_model_list[e_i] for e_i in error_preds]), -1])
                    else:
                        explain_pred_list.append(['', 1])
                return explain_pred_list
            else:
                threshold_ = - len(self.samples_iforest) * 0.6  # 有80% 判断为-1，则认为-1
                pred_list = [-1 if sum(all_is_inlier[:, i]) <= threshold_ else 1 for i in range(data_size)]
                return [[MOXINGPANDINGYICHANG_MSG, pred] for pred in pred_list]

    def predict(self, X, decision=False):
        """
        对输入的文本或文本列表进行预测是否异常；
        :param X:
        :param decision: 若为真，则返回对应的异常得分；正常返回0，异常返回(0,100]之间的数值
        :return:
        """

        def not_train(X):
            if isinstance(X, (int, float)):
                return {
                    "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                    "anomaly_msg": FEIWENBENLEIXINGSHUJU_MSG,
                    "status_code": 0
                }
            if isinstance(X, str):
                X = self.text_standard(X)
                text_len = len(X)
                not_gb2312_han_count = len([t for t in X if t not in GB2312_HAN_SET])
                if not X or not X.strip() or any(
                                k == X.strip() for k in ['不详', '未知', '空', '-', '\\N', '未填写', 'NULL', 'null', 'NaN', 'nan']) or re.search(
                    RE_ALL_IS_XING, X):
                    return {
                        "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                        "anomaly_msg": SHURUWEIQUESHIZHI_MSG,
                        "status_code": 0
                    }

                elif not self.punctuation_too_many and ((text_len > 20 and not_gb2312_han_count >= text_len * 0.6) \
                                        or (20 >= text_len > 10 and not_gb2312_han_count >= text_len * 0.8) \
                                        or (text_len <= 10 and not_gb2312_han_count > text_len * 0.9)
                                        ):
                    # 超过20个字符，若有一半非常见字；超过10个字符，若有80%非常见字；小于10个字符，若全部都是非常见字
                    return {
                        "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                        "anomaly_msg": LUANMALVGAO_MSG,
                        "status_code": 0
                    }
                elif text_len > 20 and len(remove_tags(X)) / text_len < 0.60:
                    return {
                        "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                        "anomaly_msg": HTMLGUODUO_MSG,
                        "status_code": 0
                    }
                elif not self.punctuation_too_many and (len(re.findall(re_is_punctuation, X)) / (len(re.findall(re_is_not_punctuation, X))+1) > 1.2):
                    return {
                        "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                        "anomaly_msg": TESHUFUHAOGUODUO_MSG,
                        "status_code": 0
                    }

            if self.iforest is None and self.model_error and self.enum_func is None:
                return {
                    "anomaly_detection": False,  # 是否存在文本异常，若是为真，否则为假；
                    "anomaly_msg": self.model_error,
                    "status_code": 1
                }

            elif self.iforest is None:
                return {
                    "anomaly_detection": False,  # 是否存在文本异常，若是为真，否则为假；
                    "anomaly_msg": MOXINGWEIXUNLIAN_MSG,
                    "status_code": 1
                }
            return {}

        if decision:
            if not isinstance(X, (list, tuple)):
                X = [X]
            if not self.enum_func is None:
                return [1 if self.enum_func(text) else 0 for text in X]

            text_features = [
                get_text_features(self.text_standard("{}".format(text)), self.p25_words, self.p75_words, self.words_count) for
                text in X]
            pred_list = self.samples_predict(text_features, decision=True)
            # return [0 if score >= threshold_ else round(100*(threshold_-score)/(threshold_+0.5), 3) for score, threshold_ in pred_list]
            return [0 if score >= threshold_ else [score, threshold_] for score, threshold_ in pred_list]
            # return [0 if score >= threshold_ else round((-100*(threshold_-score)/score)*0.5+(1+threshold_/0.5)*0.5, 3) for score, threshold_ in pred_list]

        not_train_ret = not_train(X)

        def normed_feature(text_features):
            normed_feature_list = []
            for features in text_features:
                normed_feat = [(1 if feat == self.all_features_min[index] else 0) if self.all_features_max[index] ==
                                                                                     self.all_features_min[index] else (
                                                                                                                       feat -
                                                                                                                       self.all_features_min[
                                                                                                                           index]) / (
                                                                                                                       self.all_features_max[
                                                                                                                           index] -
                                                                                                                       self.all_features_min[
                                                                                                                           index])
                               for index, feat in enumerate(features)]
                normed_feature_list.append(normed_feat)
            return normed_feature_list

        if not_train_ret:
            return not_train_ret

        elif isinstance(X, str):
            X = self.text_standard(X)
            if not self.enum_func is None:
                if self.enum_func(X):
                    is_inlier = -1
                    anomaly_explain = CHANGDUYICHANG_MSG
                else:
                    is_inlier = 1
                    anomaly_explain = ''
            else:
                text_features = [get_text_features(X, self.p25_words, self.p75_words, self.words_count)]
                # anomaly_explain = [explain for explain, _ in anomaly_explain_text_features]
                if self.normed:
                    text_features = normed_feature(text_features)
                anomaly_explain, is_inlier = self.samples_predict(text_features)[0]
            return {
                "anomaly_detection": True if is_inlier == -1 else False,  # 是否存在文本异常，若是为真，否则为假；
                "anomaly_msg": anomaly_explain if is_inlier == -1 else '',
                "status_code": 0
            }

        elif isinstance(X, (list, tuple, np.ndarray)):
            not_train_ret_list = [not_train(t) for t in X]
            train_ret_index = [index for index, ret in enumerate(not_train_ret_list) if not ret]
            if not self.enum_func is None:
                pred_list = [[CHANGDUYICHANG_MSG, -1 if self.enum_func(text) else 1] for text in X]
            else:
                text_features = [
                    get_text_features(self.text_standard(X[index]), self.p25_words, self.p75_words, self.words_count) for
                    index in train_ret_index]
                # anomaly_explain = [explain for explain, _ in anomaly_explain_text_features]

                if self.normed:
                    text_features = normed_feature(text_features)
                pred_list = self.samples_predict(text_features)
            for index, (anomaly_explain, is_inlier) in zip(train_ret_index, pred_list):
                not_train_ret_list[index] = {
                    "anomaly_detection": True if is_inlier == -1 else False,  # 是否存在文本异常，若是为真，否则为假；
                    "anomaly_msg": anomaly_explain if is_inlier == -1 else '',
                    "status_code": 0,
                }
            anomaly_num = sum([1 for d in not_train_ret_list if d.get('anomaly_detection')])
            # print('异常数据：{}，总数据：{}'.format(anomaly_num, len(not_train_ret_list)))
            score = 100 if anomaly_num / len(not_train_ret_list) > 0.05 else (
            100 * anomaly_num / (len(not_train_ret_list) * 0.05))
            return {"datas": not_train_ret_list,
                    "score": score,  # 错误分数，错误率越高分值越大，若错误率高于5%则100分，若错误率为0则为0分；
                    }

        else:
            return {
                "anomaly_detection": True,  # 是否存在文本异常，若是为真，否则为假；
                "anomaly_msg": SHURUSHUJULEIXINGCUOWU_MSG,
                "status_code": 0
            }

    def one_predict(self, text, lang='pinyin'):
        """
        输入单个文本，返回异常原因，若不是异常，则返回空字符串；
        :param text:
        :param lang: 若lang参数为pinyin,则返回异常对应的拼音，否则返回中文描述；
        :return:
        """
        try:
            if not isinstance(text, str) or text in NULL_SET or not text.strip():
                return ""
            anomaly_detail = self.predict(text)
            if anomaly_detail.get('anomaly_detection'):
                anomaly_msg = anomaly_detail.get('anomaly_msg', WENBENSHUJUYICHANG_MSG)
                if lang == 'pinyin':
                    anomaly_msg = ANOMALY_TO_PINYIN.get(anomaly_msg, 'error')
                return anomaly_msg
        except Exception as e:
            print('文本类型异常检测出错：{}, 错误详情：{}'.format(e, traceback.format_exc()))
        return ''

    @staticmethod
    def get_anomaly_score(return_result, count=5e5, score_span=0.01, bins=100):
        """
        计算异常分数
        错误分数，错误率越高分值越大，若错误率高于 lower_limit 则100分，若错误率为0则为0分；
        :param datas: 异常数据，[["样本1", "异常原因1"],["样本2", "异常原因2"]]
        :param count: 总测试样本，默认50w;
        :param lower_limit: 异常下限，若异常比例高过该值，则得分100；
        :return:
        """
        invaild_amount = return_result['None_count']
        err_amount = return_result['err_count']
        vaild_amount = count - invaild_amount

        table_result_dict = dict()
        table_result_dict['null_value'] = invaild_amount / count * 100
        table_result_dict['abnormal_rate'] = err_amount / vaild_amount * 100 if vaild_amount != 0 else 0
        table_result_dict['s_field'] = bins if err_amount / vaild_amount > score_span else int(
            bins * err_amount / (vaild_amount * score_span)) if vaild_amount != 0 else 0

        table_result_dict['err_example'] = return_result['err_example']
        table_result_dict['right'] = return_result['right_example']
        return table_result_dict


def main():
    import time
    mtod = ModelTextOutlierDetectionClass()
    input_file = rf'D:\Users\{USERNAME}\data\client_name_500w.txt'
    # input_file = rf'D:\Users\{USERNAME}\data\comments.txt'
    # input_file = rf'D:\Users\{USERNAME}\data\city.txt'
    with open(input_file, encoding='utf-8')as f:
        D_X = [t.strip() for t in f.readlines()][1:]
    # X = random.sample(D_X, 12000)
    # X = D_X[:12000]
    # if len(D_X)>5e5:
    random.seed(1234)
    X = random.sample(D_X, min(500000, len(D_X)))
    # X = D_X[:500000]
    print('开始训练')
    mtod.fit(X)

    print('开始预测')
    start_time = time.time()
    result_datas = []
    for text in ['好好学习，天天向上', 'hello', '腾讯科技', '&%$#(', '南山区9部王芳组', '张艳', '张丽', '张伟', '王红', '李丽', '张红']:
        ret = mtod.one_predict(text, lang='chi')
        print(text, ret)
        if ret:
            result_datas.append([text, ret])

    pred_list = mtod.predict(X[:1000], decision=False)
    pred_list = pred_list.get('datas', [])
    test_x = []
    for t, d in zip(X[:1000], pred_list):
        # if d != 0:
        #     print(t, d)
        if d.get('anomaly_detection'):
            print(t, d.get('anomaly_msg'))
            test_x.append(t)
    print('预测耗时：{}'.format(time.time() - start_time))

    df_all_features = pd.DataFrame([get_text_features(text, mtod.p25_words, mtod.p75_words, mtod.words_count) for text in X if text.strip()])
    df_all_features_mean_std = [[mean -3*std, mean+ 3*std] for mean, std in zip(df_all_features.mean(), df_all_features.std())]
    for text in test_x:
        features = get_text_features(text, mtod.p25_words, mtod.p75_words, mtod.words_count)
        for index, (min_, max_) in enumerate(df_all_features_mean_std):
            if not min_ < features[index] < max_:
                print(text, ANOMALY_EXPLAIN_FIELDS[index], min_, max_, features[index])


if __name__ == '__main__':
    main()
