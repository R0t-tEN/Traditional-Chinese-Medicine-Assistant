

import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from collections import defaultdict
import os

# --- 1. 全局配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 't2.pth'



class_names = [
    "aiye", "ajiao", "baibiandou", "baibu", "baifan", "baihe", "baihuasheshecao",
    "baikou", "baimaogen", "baishao", "baitouweng", "baizhu", "baiziren",
    "bajitian", "banlangen", "banxia", "beishashenkuai", "beishashentiao", "biejia",
    "cangzhu", "caoguo", "caokou", "cebaiye", "chaihu", "chantui", "chenpi",
    "chenxiang", "chishao", "chishizhi", "chongcao", "chuanshanjia", "chuanxinlian",
    "cishi", "dafupi", "dangshen", "danshen", "daqingye", "daxueteng", "digupi",
    "dilong", "diyu", "duzhong", "fangfeng", "foshou", "fuling", "fupenzi", "fuzi",
    "gancao", "ganjiang", "gegen", "gouqizi", "gouteng", "guanzhong", "guya",
    "hehuanpi", "heshouwu", "honghua", "hongkou", "houpu", "huaihua", "huangbo",
    "huangjing", "huangqin", "huomaren", "huzhang", "jiangcan", "jianghuang",
    "jineijin", "jingjie", "jinqiancao", "jinyinhua", "jixueteng", "juemingzi",
    "kushen", "laifuzi", "lianqiao", "lianzixin", "lingzhi", "lizhihe", "longgu",
    "lulutong", "luohanguo", "luoshiteng", "maidong", "maiya", "mohanlian",
    "mudanpi", "muli", "muxiang", "niuxi", "nvzhenzi", "paojiang", "peilan",
    "pugongying", "puhuang", "qianghuo", "qianhu", "qinghao", "quanxie", "renshen",
    "renshenqiepian", "roucongronggen", "roucongrongpian", "roudoukou", "rougui",
    "sangpiaoxiao", "sangshen", "sanqi", "shanyao", "shanzha", "shanzhuyu",
    "sharen", "shechuangzi", "shegan", "shengma", "shenqu", "shichangpu", "shigao",
    "shihu", "shouwutengkuai", "shouwutengpian", "shuihonghuazi", "shuiniujiao",
    "suanzaoren", "taoren", "tiandong", "tiankuizi", "tianmakuai", "tianmapian",
    "tiannanxing", "tongcao", "tubiechong", "tusizi", "wujiapi", "wulingzhi",
    "wumei", "wuweizi", "xiakucao", "xiangfu", "xianhecao", "xiaohuixiang",
    "xinyi", "xixin", "xuduan", "yejuhua", "yimucao", "yinchen", "yiyiren",
    "yuanzhi", "yujin", "yuzhupian", "yuzhutiao", "zelan", "zhebeimu",
    "zhenzhumu", "zhimu", "zhiqiaopian", "zhiqiaotiao", "zhishi", "zhuru", "zicao",
    "zihuadiding","ziyuan"
]

# 动态获取类别数量，确保与列表一致 (现在是163)
NUM_CLASSES = len(class_names)

# 拼音到中文的映射字典
pinyin_to_chinese = {
    "aiye": "艾叶", "ajiao": "阿胶", "baibiandou": "白扁豆", "baibu": "百部", "baifan": "白矾",
    "baihe": "百合", "baihuasheshecao": "白花蛇舌草", "baikou": "白蔻", "baimaogen": "白茅根",
    "baishao": "白芍", "baitouweng": "白头翁", "baizhu": "白术", "baiziren": "柏子仁",
    "bajitian": "巴戟天", "banlangen": "板蓝根", "banxia": "半夏", "beishashenkuai": "北沙参",
    "beishashentiao": "北沙参", "biejia": "鳖甲", "cangzhu": "苍术", "caoguo": "草果",
    "caokou": "草豆蔻", "cebaiye": "侧柏叶", "chaihu": "柴胡", "chantui": "蝉蜕",
    "chenpi": "陈皮", "chenxiang": "沉香", "chishao": "赤芍", "chishizhi": "赤石脂",
    "chongcao": "虫草", "chuanshanjia": "穿山甲", "chuanxinlian": "穿心莲", "cishi": "磁石",
    "dafupi": "大腹皮", "dangshen": "党参", "danshen": "丹参", "daqingye": "大青叶",
    "daxueteng": "大血藤", "digupi": "地骨皮", "dilong": "地龙", "diyu": "地榆",
    "duzhong": "杜仲", "fangfeng": "防风", "foshou": "佛手", "fuling": "茯苓", "fupenzi": "覆盆子",
    "fuzi": "附子", "gancao": "甘草", "ganjiang": "干姜", "gegen": "葛根", "gouqizi": "枸杞子",
    "gouteng": "钩藤", "guanzhong": "贯众", "guya": "谷芽", "hehuanpi": "合欢皮",
    "heshouwu": "何首乌", "honghua": "红花", "hongkou": "红豆蔻", "houpu": "厚朴", "huaihua": "槐花",
    "huangbo": "黄柏", "huangjing": "黄精", "huangqin": "黄芩", "huomaren": "火麻仁",
    "huzhang": "虎杖", "jiangcan": "僵蚕", "jianghuang": "姜黄", "jineijin": "鸡内金",
    "jingjie": "荆芥", "jinqiancao": "金钱草", "jinyinhua": "金银花", "jixueteng": "鸡血藤",
    "juemingzi": "决明子", "kushen": "苦参", "laifuzi": "莱菔子", "lianqiao": "连翘",
    "lianzixin": "莲子心", "lingzhi": "灵芝", "lizhihe": "荔枝核", "longgu": "龙骨",
    "lulutong": "路路通", "luohanguo": "罗汉果", "luoshiteng": "络石藤", "maidong": "麦冬",
    "maiya": "麦芽", "mohanlian": "墨旱莲", "mudanpi": "牡丹皮", "muli": "牡蛎",
    "muxiang": "木香", "niuxi": "牛膝", "nvzhenzi": "女贞子", "paojiang": "炮姜",
    "peilan": "佩兰", "pugongying": "蒲公英", "puhuang": "蒲黄", "qianghuo": "羌活",
    "qianhu": "前胡", "qinghao": "青蒿", "quanxie": "全蝎", "renshen": "人参",
    "renshenqiepian": "人参切片", "roucongronggen": "肉苁蓉", "roucongrongpian": "肉苁蓉",
    "roudoukou": "肉豆蔻", "rougui": "肉桂", "sangpiaoxiao": "桑螵蛸", "sangshen": "桑葚",
    "sanqi": "三七", "shanyao": "山药", "shanzha": "山楂", "shanzhuyu": "山茱萸",
    "sharen": "砂仁", "shechuangzi": "蛇床子", "shegan": "射干", "shengma": "升麻",
    "shenqu": "神曲", "shichangpu": "石菖蒲", "shigao": "石膏", "shihu": "石斛",
    "shouwutengkuai": "首乌藤", "shouwutengpian": "首乌藤", "shuihonghuazi": "水红花子",
    "shuiniujiao": "水牛角", "suanzaoren": "酸枣仁", "taoren": "桃仁", "tiandong": "天冬",
    "tiankuizi": "天葵子", "tianmakuai": "天麻", "tianmapian": "天麻", "tiannanxing": "天南星",
    "tongcao": "通草", "tubiechong": "土鳖虫", "tusizi": "菟丝子", "wujiapi": "五加皮",
    "wulingzhi": "五灵脂", "wumei": "乌梅", "wuweizi": "五味子", "xiakucao": "夏枯草",
    "xiangfu": "香附", "xianhecao": "仙鹤草", "xiaohuixiang": "小茴香", "xinyi": "辛夷",
    "xixin": "细辛", "xuduan": "续断", "yejuhua": "野菊花", "yimucao": "益母草",
    "yinchen": "茵陈", "yiyiren": "薏苡仁", "yuanzhi": "远志", "yujin": "郁金",
    "yuzhupian": "玉竹", "yuzhutiao": "玉竹", "zelan": "泽兰", "zhebeimu": "浙贝母",
    "zhenzhumu": "珍珠母", "zhimu": "知母", "zhiqiaopian": "枳壳", "zhiqiaotiao": "枳壳",
    "zhishi": "枳实", "zhuru": "竹茹", "zicao": "紫草", "zihuadiding": "紫花地丁","ziyuan":"紫菀"
}

# --- 2. 模型加载 ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
# 这里的 NUM_CLASSES 现在是 163，与模型匹配
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

try:
    print(f"正在从'{MODEL_PATH}'加载模型...")
    # 严格模式设为 True (默认)，确保键完全匹配
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
    model.to(DEVICE)
    model.eval()
    print("模型加载成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 '{MODEL_PATH}'。程序将无法进行预测。")
    model = None
except Exception as e:
    print(f"加载模型时发生严重错误: {e}")
    print("请确保 'class_names' 列表的长度 (现在是 {NUM_CLASSES}) 与训练模型时完全一致。")
    model = None

# --- 3. 核心预测函数 ---
def predict_herb(input_image):
    if model is None:
        return None, "错误：模型未能成功加载，无法进行识别。请检查启动时的报错信息。"
    if input_image is None:
        return None, "请先上传一张图片再点击识别按钮。"

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = preprocess(input_image)
    batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
    
    with torch.no_grad():
        out = model(batch_t)
    
    probabilities = torch.nn.functional.softmax(out, dim=1)[0]
    
    # ### 修正: 解决中文名重复导致概率被覆盖的问题
    # 使用 defaultdict 来聚合具有相同中文名的类别的概率
    aggregated_confidences = defaultdict(float)
    for i, prob in enumerate(probabilities):
        pinyin_name = class_names[i]
        chinese_name = pinyin_to_chinese.get(pinyin_name, pinyin_name)
        aggregated_confidences[chinese_name] += prob.item()

    # ### 修正: 基于聚合后的结果生成最终输出，确保两个输出框一致
    if not aggregated_confidences:
         return None, "识别失败，无法计算置信度。"

    # 从聚合后的结果中找到最高置信度的类别
    top_pred_name = max(aggregated_confidences, key=aggregated_confidences.get)
    top_pred_prob = aggregated_confidences[top_pred_name]
    
    result_text = f"识别结果: {top_pred_name}\n置信度: {top_pred_prob:.2%}"
    
    # 返回聚合后的字典和格式化文本
    return aggregated_confidences, result_text



# --- 4. Gradio 界面布局 ---
with gr.Blocks(theme=gr.themes.Soft(), title="中草药识别器") as iface:
    gr.Markdown("# 智能中草药识别器")
    gr.Markdown("上传一张中草药的图片，AI会尝试识别它的种类。")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传药材图片")
            submit_btn = gr.Button("开始识别", variant="primary")
        
        with gr.Column(scale=1):
            prob_output = gr.Label(num_top_classes=5, label="识别概率 (Top 5)")
            text_output = gr.Textbox(label="识别结果")

    # --- 5. 事件绑定 ---
    submit_btn.click(
        fn=predict_herb,
        inputs=image_input,
        outputs=[prob_output, text_output]
    )
    
    # 示例部分，如果 'examples' 文件夹存在，则加载示例
    example_list = []
    if os.path.isdir("examples"):
        example_list = [os.path.join("examples", f) for f in os.listdir("examples")]

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=image_input,
            outputs=[prob_output, text_output],
            fn=predict_herb,
            cache_examples=True # 建议开启，可以缓存示例结果，加快点击速度
        )


# --- 6. 启动应用 ---
if __name__ == '__main__':
    print("正在启动Gradio Web界面...")
    # 使用 share=True 来创建一个可分享的公共链接
    iface.launch(share=True)