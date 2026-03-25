#!/usr/bin/env python
"""
Electronics Product Data Generator.

Generates synthetic FAQ and document data for electronics product knowledge base.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class FAQItem:
    """FAQ item structure."""

    id: str
    question: str
    answer: str
    category: str
    keywords: list[str] = field(default_factory=list)
    product: Optional[str] = None
    brand: Optional[str] = None


@dataclass
class DocumentItem:
    """Document item structure."""

    id: str
    title: str
    content: str
    category: str
    product: Optional[str] = None
    brand: Optional[str] = None
    keywords: list[str] = field(default_factory=list)


# Brand data
BRANDS = {
    "手机": ["Apple", "华为", "小米", "OPPO", "vivo", "三星", "荣耀", "一加", "realme", "魅族"],
    "笔记本": ["Apple", "华为", "小米", "联想", "戴尔", "惠普", "华硕", "宏碁", "微软", "雷神"],
    "平板": ["Apple", "华为", "小米", "三星", "联想", "荣耀", "OPPO", "vivo"],
    "耳机": ["Apple", "华为", "小米", "索尼", "Bose", "森海塞尔", "OPPO", "vivo", "漫步者", "JBL"],
    "智能手表": ["Apple", "华为", "小米", "三星", "OPPO", "vivo", "荣耀", "佳明"],
    "智能家居": ["小米", "华为", "天猫精灵", "小度", "海尔", "美的", "萤石", "TP-Link"],
    "配件": ["Apple", "安克", "小米", "华为", "倍思", "绿联", "罗马仕", "紫米"],
}

# Product models
PRODUCTS = {
    "Apple": {
        "手机": ["iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15 Plus", "iPhone 15", "iPhone 14", "iPhone SE"],
        "笔记本": ["MacBook Pro 16寸", "MacBook Pro 14寸", "MacBook Air 15寸", "MacBook Air 13寸"],
        "平板": ["iPad Pro 12.9寸", "iPad Pro 11寸", "iPad Air", "iPad", "iPad mini"],
        "耳机": ["AirPods Pro 2", "AirPods 3", "AirPods Max", "Beats Studio Pro"],
        "智能手表": ["Apple Watch Ultra 2", "Apple Watch Series 9", "Apple Watch SE 2"],
    },
    "华为": {
        "手机": ["Mate 60 Pro+", "Mate 60 Pro", "Mate 60", "P60 Pro", "P60", "nova 12 Pro", "nova 12"],
        "笔记本": ["MateBook X Pro", "MateBook 14", "MateBook D16", "MateBook D14"],
        "平板": ["MatePad Pro 13.2", "MatePad Pro 11", "MatePad Air", "MatePad 11"],
        "耳机": ["FreeBuds Pro 3", "FreeBuds 5", "FreeBuds SE 2", "FreeClip"],
        "智能手表": ["Watch GT 4", "Watch 4 Pro", "Watch GT 3", "Watch Ultimate"],
    },
    "小米": {
        "手机": ["Xiaomi 14 Ultra", "Xiaomi 14 Pro", "Xiaomi 14", "Redmi K70 Pro", "Redmi K70", "Redmi Note 13 Pro+"],
        "笔记本": ["小米笔记本Pro 16", "小米笔记本Pro 15", "RedmiBook Pro 15", "RedmiBook 14"],
        "平板": ["Xiaomi Pad 6 Pro", "Xiaomi Pad 6", "Redmi Pad SE"],
        "耳机": ["Xiaomi Buds 4 Pro", "Xiaomi Buds 4", "Redmi Buds 5 Pro", "Redmi Buds 5"],
        "智能手表": ["Xiaomi Watch S3", "Xiaomi Watch 2 Pro", "Redmi Watch 4"],
    },
    "OPPO": {
        "手机": ["Find X7 Ultra", "Find X7 Pro", "Find X7", "Reno 11 Pro+", "Reno 11", "一加 12"],
        "耳机": ["OPPO Enco X2", "OPPO Enco Air 3", "一加 Buds Pro 2"],
        "智能手表": ["OPPO Watch X", "OPPO Watch 3 Pro", "OPPO Watch 3"],
    },
    "vivo": {
        "手机": ["X100 Pro+", "X100 Pro", "X100", "S18 Pro", "S18", "iQOO 12 Pro"],
        "耳机": ["vivo TWS 4", "vivo TWS 3 Pro", "iQOO TWS 1"],
        "智能手表": ["vivo Watch 3", "vivo Watch 2"],
    },
    "三星": {
        "手机": ["Galaxy S24 Ultra", "Galaxy S24+", "Galaxy S24", "Galaxy Z Fold5", "Galaxy Z Flip5", "Galaxy A54"],
        "平板": ["Galaxy Tab S9 Ultra", "Galaxy Tab S9+", "Galaxy Tab S9", "Galaxy Tab A9+"],
        "耳机": ["Galaxy Buds3 Pro", "Galaxy Buds2 Pro", "Galaxy Buds FE"],
        "智能手表": ["Galaxy Watch6 Classic", "Galaxy Watch6", "Galaxy Watch5 Pro"],
    },
}

# FAQ templates by category
FAQ_TEMPLATES = {
    "手机操作": [
        ("{product}如何截屏？", "{brand}{product}截屏方法：同时按住电源键和音量加键，屏幕闪烁即表示截屏成功。截图会自动保存到相册的截屏文件夹中。也可以使用控制中心的截屏按钮或三指下滑截屏（需在设置中开启）。"),
        ("{product}如何录屏？", "{brand}{product}录屏方法：从屏幕顶部下拉打开控制中心，点击屏幕录制按钮开始录制。录制过程中状态栏会显示红色录制指示。再次点击即可停止录制。录制视频保存在相册中。"),
        ("{product}如何开启深色模式？", "{brand}{product}开启深色模式：进入设置 > 显示与亮度 > 选择深色模式。也可以从控制中心快速切换。深色模式可以减少眼睛疲劳，同时在OLED屏幕上还能省电。"),
        ("{product}如何设置面部解锁？", "{brand}{product}面部解锁设置：进入设置 > 安全与隐私 > 面部识别 > 添加面部数据。按照提示将面部对准识别框，完成录入后即可使用面部解锁。建议同时设置备用解锁方式。"),
        ("{product}如何分屏操作？", "{brand}{product}分屏操作方法：打开多任务界面，长按应用卡片选择分屏模式，然后选择第二个应用。部分应用支持悬浮窗模式。分屏时可以调整两个应用的显示比例。"),
        ("{product}如何转移数据到新手机？", "{brand}{product}数据迁移：可使用官方换机助手应用，通过无线连接或数据线将旧手机数据迁移到新手机。支持迁移通讯录、短信、照片、应用等数据。迁移过程通常需要30分钟到2小时。"),
    ],
    "手机电池": [
        ("{product}电池续航多久？", "{brand}{product}配备大容量电池，正常使用可续航1-2天。支持快充技术，约30分钟可充至50%电量。开启省电模式可延长续航时间。建议避免长时间高温环境下充电。"),
        ("{product}如何延长电池寿命？", "{brand}{product}电池保养建议：1.避免电量低于20%再充电；2.避免长时间高温环境使用；3.使用原装充电器；4.避免边玩边充；5.定期清理后台应用。这些习惯可以有效延长电池使用寿命。"),
        ("{product}充电发烫正常吗？", "{brand}{product}快充时轻微发热属于正常现象。如果温度过高，建议：1.取下手机壳散热；2.避免边充边玩大型游戏；3.使用原装充电器；4.在通风良好的环境充电。如发热严重请联系售后。"),
        ("{product}支持无线充电吗？", "{brand}{product}支持{power}W无线充电，需使用Qi认证无线充电器。无线充电比有线充电速度稍慢，但更加便捷。建议使用官方或认证的无线充电器，避免使用劣质产品损坏设备。"),
        ("{product}充电速度是多少？", "{brand}{product}支持{power}W有线快充，使用原装充电器可在约{time}分钟充满。实际充电速度会因温度、电量等因素有所差异。建议使用原装充电器和数据线以获得最佳充电体验。"),
    ],
    "手机相机": [
        ("{product}相机像素是多少？", "{brand}{product}配备{main}MP主摄 + {ultra}MP超广角 + {tele}MP长焦三摄系统。支持OIS光学防抖，夜景模式，8K视频录制。前置{front}MP摄像头，支持4K视频通话和多种美颜滤镜。"),
        ("{product}如何拍出好看的夜景照片？", "{brand}{product}夜景拍摄技巧：1.使用夜景模式自动优化；2.保持手机稳定或使用三脚架；3.点击对焦后等待曝光稳定；4.避免光源直射镜头。夜景模式会自动多帧合成，拍摄时间约3-5秒。"),
        ("{product}如何使用专业模式拍照？", "{brand}{product}专业模式参数设置：ISO控制感光度（白天100-400，夜晚800-3200）；快门速度控制曝光时间；白平衡调整色温；对焦模式选择自动或手动。建议从RAW格式拍摄以便后期调色。"),
        ("{product}支持多少倍变焦？", "{brand}{product}支持{digital}倍数码变焦，{optical}倍光学变焦，{hybrid}倍混合变焦。光学变焦画质最佳，混合变焦次之，数码变焦会有一定画质损失。建议优先使用光学变焦范围。"),
        ("{product}如何拍摄慢动作视频？", "{brand}{product}慢动作拍摄：打开相机切换到慢动作模式，支持{fps}fps慢动作录制。拍摄时光线要充足，保持手机稳定。录制后可在相册中编辑慢动作的开始和结束时间点。"),
    ],
    "笔记本性能": [
        ("{product}配置参数是什么？", "{brand}{product}配置：搭载{cpu}处理器，{ram}GB统一内存，{storage}GB SSD固态硬盘。配备{screen}英寸{resolution}分辨率屏幕，支持{refresh}Hz刷新率。续航约{battery}小时。"),
        ("{product}适合游戏吗？", "{brand}{product}搭载{gpu}显卡，可以流畅运行大部分主流游戏。3A大作建议中高画质运行，电竞类游戏支持高帧率。如需更好游戏体验，建议外接显示器和散热底座。"),
        ("{product}如何提升运行速度？", "{brand}{product}性能优化建议：1.定期清理系统缓存；2.关闭开机自启动应用；3.保持至少20%磁盘空间；4.及时更新系统和驱动；5.使用SSD而非机械硬盘存储。"),
        ("{product}续航时间多长？", "{brand}{product}日常办公续航约{battery}小时，视频播放可达{video}小时。续航时间受亮度、应用负载等因素影响。开启省电模式可延长约30%续航。建议外出携带充电器。"),
        ("{product}支持外接显示器吗？", "{brand}{product}支持通过USB-C或HDMI接口外接显示器，最高支持{external}分辨率。可扩展为镜像显示或扩展桌面模式。建议使用认证的转接头和线缆以获得稳定连接。"),
    ],
    "平板使用": [
        ("{product}支持手写笔吗？", "{brand}{product}支持官方手写笔，具有{pressure}级压感，支持倾斜感应。手写笔延迟仅{latency}ms，书写体验接近真实纸笔。适合绘画、记笔记、批注文档等场景使用。"),
        ("{product}可以连接键盘吗？", "{brand}{product}支持蓝牙键盘和官方键盘保护套。连接后可使用快捷键提升效率。部分型号支持触控板手势。配合键盘和平板模式，可作为轻薄笔记本替代品。"),
        ("{product}适合绘画吗？", "{brand}{product}配备{screen}英寸高色域屏幕，支持P3广色域和原彩显示。配合手写笔使用，非常适合数字绘画。建议搭配绘画软件如Procreate或概念画板使用。"),
        ("{product}如何分屏多任务？", "{brand}{product}分屏方法：从屏幕边缘向内滑动停顿，调出应用侧边栏，拖动应用到屏幕一侧实现分屏。支持调整分屏比例，也可以使用悬浮窗口模式同时运行多个应用。"),
    ],
    "耳机音质": [
        ("{product}音质怎么样？", "{brand}{product}采用{driver}mm动圈单元，支持{codec}高清音频编解码，频率响应范围{freq}Hz。经过专业调音，三频均衡，低音下潜深，高音通透。支持EQ自定义调节。"),
        ("{product}降噪效果如何？", "{brand}{product}支持{anc}dB主动降噪，可有效隔绝环境噪音。提供多种降噪模式：深度降噪适合嘈杂环境，通透模式可听清周围声音。降噪不影响音质表现。"),
        ("{product}续航多久？", "{brand}{product}耳机单次续航{earphone}小时，配合充电盒总续航达{total}小时。支持快充：充电{quick_charge}分钟可使用{quick_time}小时。开启降噪后续航会有所降低。"),
        ("{product}如何连接手机？", "{brand}{product}支持蓝牙{version}，首次使用打开充电盒盖，靠近手机会自动弹出配对提示。支持双设备连接，可在手机和电脑间无缝切换。连接稳定，延迟低于{latency}ms。"),
        ("{product}佩戴舒适吗？", "{brand}{product}单耳重量仅{weight}g，采用人体工学设计，提供{tips}种尺寸耳塞。长时间佩戴不易疲劳。IPX{waterproof}级防水，运动出汗也不用担心。"),
    ],
    "智能手表": [
        ("{product}能测心率吗？", "{brand}{product}配备光学心率传感器，支持24小时连续心率监测。心率异常时会自动提醒。还支持血氧饱和度检测、睡眠监测、压力监测等健康功能。数据可在手机App中查看。"),
        ("{product}续航多久？", "{brand}{product}智能模式下续航{smart}天，长续航模式可达{long}天。支持无线充电，约{charge}小时充满。续航受使用频率、功能开启状态影响。建议夜间充电。"),
        ("{product}可以接电话吗？", "{brand}{product}支持蓝牙通话功能，手表内置扬声器和麦克风。来电时可直接接听或拒接。配合eSIM版本可独立使用，无需手机在身边也能接打电话。"),
        ("{product}支持哪些运动模式？", "{brand}{product}支持{sports}种运动模式，包括跑步、游泳、骑行、登山等。内置GPS可独立记录运动轨迹。支持运动数据实时显示和语音播报，运动后自动生成详细报告。"),
        ("{product}防水等级？", "{brand}{product}具备{waterproof}米防水等级，可用于游泳和浅水活动。但不建议潜水或热水浴时佩戴。接触海水或泳池水后，建议用清水冲洗并擦干。"),
    ],
    "智能家居": [
        ("{product}如何连接WiFi？", "{brand}{product}连接方法：下载官方App，登录后选择添加设备，按照提示连接WiFi。需使用2.4GHz WiFi网络，确保设备处于配对模式。连接失败可尝试重置设备或切换网络。"),
        ("{product}支持语音控制吗？", "{brand}{product}支持主流语音助手控制，可通过语音指令开关设备、调节亮度、设置场景等。配合智能音箱使用体验更佳。可在App中自定义语音指令和场景联动。"),
        ("{product}可以远程控制吗？", "{brand}{product}支持远程控制，只要有网络就可以通过手机App控制设备。支持定时开关、场景联动、设备分享等功能。出差旅行时也能管理家中智能设备。"),
    ],
    "配件充电": [
        ("{product}输出功率是多少？", "{brand}{product}支持{power}W快充输出，配备{port}个充电接口。支持PD3.0、QC4+等快充协议，兼容主流手机、平板、笔记本。智能分配功率，保护设备安全。"),
        ("{product}充电宝容量多大？", "{brand}{product}电池容量{capacity}mAh，可给普通手机充电{times}次左右。支持双向快充，自身充满约{self_charge}小时。航空随身携带需注意容量限制（通常不超过20000mAh）。"),
        ("{product}数据线多长？", "{brand}{product}线长{length}米，采用{material}材质外皮，耐弯折。支持{power}W快充和{speed}数据传输。经过{bends}次弯折测试，使用寿命长。"),
    ],
}

# Document templates
DOCUMENT_TEMPLATES = {
    "使用教程": [
        {
            "title": "{product}快速入门指南",
            "sections": [
                "一、开箱检查\n\n打开包装盒，请确认以下物品齐全：\n- {product}主机 x1\n- 充电器 x1\n- 数据线 x1\n- 说明书 x1\n- 保修卡 x1\n\n如有遗漏，请及时联系销售商。",
                "二、首次开机\n\n1. 长按电源键3秒开机\n2. 选择语言和地区\n3. 连接WiFi网络\n4. 登录{brand}账号\n5. 同意用户协议\n6. 设置指纹或面部解锁\n7. 数据迁移（可选）",
                "三、基本操作\n\n1. 解锁：抬起手机或按电源键唤醒，使用面部/指纹解锁\n2. 截屏：电源键+音量加键同时按下\n3. 控制中心：从屏幕顶部右侧下滑\n4. 通知中心：从屏幕顶部左侧下滑\n5. 多任务：从屏幕底部上滑停顿",
                "四、常见问题\n\nQ: 手机发热怎么办？\nA: 高负载使用时会轻微发热，属于正常现象。建议避免边充边玩，在通风环境使用。\n\nQ: 如何延长电池寿命？\nA: 避免电量过低再充电，使用原装充电器，避免高温环境。",
                "五、售后服务\n\n{brand}提供1年质保服务，质保期内非人为损坏免费维修。\n\n客服热线：400-xxx-xxxx\n在线客服：{brand}官网或App\n服务网点：可在官网查询附近服务网点",
            ],
        },
        {
            "title": "{product}相机完全指南",
            "sections": [
                "一、相机基础\n\n{brand}{product}配备专业级相机系统，主摄为{main}MP，支持OIS光学防抖。还配备{ultra}MP超广角镜头和{tele}MP长焦镜头，覆盖多种拍摄场景。",
                "二、拍摄模式\n\n1. 照片模式：自动调节参数，适合日常拍摄\n2. 人像模式：虚化背景，突出人物\n3. 夜景模式：多帧合成，提升暗光画质\n4. 专业模式：手动调节ISO、快门、白平衡\n5. 全景模式：拍摄宽广场景\n6. 慢动作：拍摄高速运动",
                "三、拍摄技巧\n\n【人像摄影】\n- 选择2x焦段效果最佳\n- 光线充足时画质更好\n- 背景简洁干净效果更好\n\n【夜景拍摄】\n- 使用三脚架保持稳定\n- 点击对焦后等待曝光调整\n- 避免强光源直射镜头",
                "四、视频录制\n\n支持最高8K/30fps视频录制，4K/60fps为推荐设置。\n\n视频防抖功能：\n- 标准防抖：适合日常拍摄\n- 增强防抖：适合运动场景\n- 运动模式：极限运动场景",
                "五、后期编辑\n\n照片编辑功能：\n- 基础调整：亮度、对比度、饱和度\n- 滤镜效果：多种预设滤镜\n- 裁剪旋转：调整构图\n- 标注工具：添加文字箭头",
            ],
        },
    ],
    "产品规格": [
        {
            "title": "{product}详细参数",
            "sections": [
                "【基本信息】\n产品名称：{product}\n品牌：{brand}\n发布时间：2024年\n\n【屏幕】\n尺寸：{screen}英寸\n类型：{screen_type}\n分辨率：{resolution}\n刷新率：{refresh}Hz\n亮度：{brightness}nit\n色彩：{color}色域",
                "【处理器】\n芯片：{chip}\n制程：{process}nm\nCPU核心：{cpu_cores}核\nGPU：{gpu}\nNPU：{npu}\n\n【内存存储】\n运行内存：{ram}GB\n存储容量：{storage}GB\n扩展存储：{expand}",
                "【相机】\n后置主摄：{main}MP\n超广角：{ultra}MP\n长焦：{tele}MP\n前置：{front}MP\n视频：最高{video}K\n防抖：{stabilization}\n\n【电池充电】\n容量：{battery}mAh\n快充：{charge}W\n无线充：{wireless}W\n续航：{endurance}小时",
                "【网络连接】\n移动网络：5G\nWiFi：WiFi 7\n蓝牙：{bluetooth}\nNFC：支持\n定位：GPS/北斗/格洛纳斯\n\n【其他功能】\n防水等级：{waterproof}\n指纹：{fingerprint}\n面部识别：支持\n红外遥控：支持",
                "【外观尺寸】\n尺寸：{dimensions}mm\n重量：{weight}g\n材质：{material}\n颜色：{colors}\n\n【包装清单】\n主机 x1\n充电器 x1\n数据线 x1\n保护壳 x1\n说明书 x1",
            ],
        },
    ],
    "故障排除": [
        {
            "title": "{product}常见问题解决方案",
            "sections": [
                "一、无法开机\n\n可能原因：\n1. 电池电量耗尽\n2. 系统死机\n3. 硬件故障\n\n解决方案：\n1. 连接充电器充电15分钟后再试\n2. 长按电源键+音量减键10秒强制重启\n3. 如仍无法开机，联系售后检测",
                "二、充电异常\n\n问题表现：\n- 充不进电\n- 充电慢\n- 充电发热\n\n解决方案：\n1. 检查充电器和数据线是否原装\n2. 清洁充电接口\n3. 更换插座尝试\n4. 避免高温环境充电\n5. 重启后重试",
                "三、WiFi连接问题\n\n问题表现：\n- 搜不到WiFi\n- 连接后断开\n- 网速慢\n\n解决方案：\n1. 重启路由器和手机\n2. 忘记网络后重新连接\n3. 检查WiFi密码是否正确\n4. 还原网络设置\n5. 更新系统版本",
                "四、应用闪退\n\n可能原因：\n1. 应用版本过旧\n2. 内存不足\n3. 系统兼容性\n\n解决方案：\n1. 更新应用到最新版本\n2. 清理后台应用释放内存\n3. 清除应用缓存\n4. 卸载重装应用\n5. 更新手机系统",
                "五、发热严重\n\n正常发热场景：\n- 高负载游戏\n- 快速充电\n- 5G网络使用\n- 环境温度高\n\n异常发热处理：\n1. 取下手机壳散热\n2. 关闭后台应用\n3. 降低屏幕亮度\n4. 使用散热背夹\n5. 如持续异常送修检测",
            ],
        },
    ],
}


def generate_faqs(count: int = 500) -> list[FAQItem]:
    """Generate FAQ items."""
    faqs = []
    faq_id = 1

    for category, templates in FAQ_TEMPLATES.items():
        # Determine category brand/product type
        if category in ["手机操作", "手机电池", "手机相机"]:
            product_type = "手机"
        elif category in ["笔记本性能"]:
            product_type = "笔记本"
        elif category in ["平板使用"]:
            product_type = "平板"
        elif category in ["耳机音质"]:
            product_type = "耳机"
        elif category in ["智能手表"]:
            product_type = "智能手表"
        elif category in ["智能家居"]:
            product_type = "智能家居"
        else:
            product_type = "配件"

        brands = BRANDS.get(product_type, ["Apple", "华为", "小米"])

        for template in templates:
            question_template, answer_template = template

            # Generate for multiple brands/products
            for brand in brands[:3]:  # Top 3 brands per category
                products = PRODUCTS.get(brand, {}).get(product_type, [f"{brand}{product_type}"])
                for product in products[:2]:  # Top 2 products per brand
                    # Fill in templates
                    question = question_template.format(product=product, brand=brand)
                    answer = answer_template.format(
                        product=product,
                        brand=brand,
                        power=random.choice([20, 30, 45, 65, 66, 67, 80, 100, 120, 150]),
                        time=random.choice([30, 40, 45, 50, 60]),
                        main=random.choice([48, 50, 64, 108, 200]),
                        ultra=random.choice([12, 13, 48, 50]),
                        tele=random.choice([5, 8, 10, 12, 48, 64]),
                        front=random.choice([12, 16, 20, 32]),
                        digital=random.choice([30, 50, 100]),
                        optical=random.choice([2, 3, 3.5, 4, 5, 10]),
                        hybrid=random.choice([5, 10, 20]),
                        fps=random.choice([120, 240, 960]),
                        cpu=random.choice(["A17 Pro", "骁龙8 Gen3", "天玑9300", "麒麟9000S"]),
                        ram=random.choice([8, 12, 16, 24]),
                        storage=random.choice([128, 256, 512, 1024]),
                        screen=random.choice([6.1, 6.3, 6.5, 6.7, 6.8]),
                        resolution=random.choice(["2K+", "FHD+", "3K"]),
                        refresh=random.choice([60, 90, 120, 144]),
                        battery=random.choice([12, 14, 16, 18, 20, 24]),
                        video=random.choice([15, 18, 20]),
                        external=random.choice(["4K 60Hz", "5K 60Hz", "6K 60Hz"]),
                        gpu=random.choice(["Adreno 750", "Mali-G720", "Apple GPU"]),
                        driver=random.choice([11, 12, 13, 14, 15]),
                        codec=random.choice(["LDAC", "aptX HD", "LHDC", "AAC"]),
                        freq="20-40000",
                        anc=random.choice([35, 40, 42, 45, 48]),
                        earphone=random.choice([5, 6, 7, 8]),
                        total=random.choice([24, 28, 30, 36, 40]),
                        quick_charge=random.choice([5, 10, 15]),
                        quick_time=random.choice([1, 2, 3]),
                        version=random.choice(["5.2", "5.3", "5.4"]),
                        latency=random.choice([40, 50, 54, 80, 100]),
                        weight=random.choice([4.5, 5.0, 5.4, 5.8, 6.0]),
                        tips=random.choice([3, 4, 5]),
                        waterproof=random.choice([4, 5, 6, 7, 8]),
                        smart=random.choice([1, 2, 3, 4]),
                        long=random.choice([7, 10, 14, 21]),
                        charge=random.choice([1, 1.5, 2]),
                        sports=random.choice([80, 100, 120, 150]),
                        pressure=random.choice([2048, 4096, 8192]),
                        capacity=random.choice([10000, 20000, 30000]),
                        times=random.choice([2, 3, 4, 5, 6]),
                        self_charge=random.choice([3, 4, 5, 6]),
                        length=random.choice([1, 1.2, 1.5, 2]),
                        material=random.choice(["编织", "TPE", "尼龙"]),
                        speed=random.choice(["480Mbps", "5Gbps", "10Gbps", "40Gbps"]),
                        bends=random.choice([10000, 20000, 30000]),
                        brightness=random.choice([1000, 1200, 1500, 2000, 2500]),
                        color=random.choice(["P3", "sRGB", "DCI-P3"]),
                        port=random.choice([2, 3, 4]),
                    )

                    # Extract keywords
                    keywords = [brand]
                    if product:
                        keywords.append(product)
                    keywords.extend([w for w in question if w.isalpha() and len(w) > 1][:3])

                    faq = FAQItem(
                        id=f"faq-{faq_id:04d}",
                        question=question,
                        answer=answer,
                        category=category,
                        keywords=list(set(keywords)),
                        product=product,
                        brand=brand,
                    )
                    faqs.append(faq)
                    faq_id += 1

                    if len(faqs) >= count:
                        return faqs

    return faqs


def generate_documents(count: int = 200) -> list[DocumentItem]:
    """Generate document items."""
    documents = []
    doc_id = 1

    for category, templates in DOCUMENT_TEMPLATES.items():
        for template in templates:
            # Generate for multiple brands/products
            for brand, brand_products in PRODUCTS.items():
                for product_type, products in brand_products.items():
                    for product in products[:1]:  # One doc per product
                        title = template["title"].format(
                            product=product,
                            brand=brand,
                        )

                        # Combine sections into content
                        content = "\n\n".join(template["sections"])

                        # Fill in placeholders
                        content = content.format(
                            product=product,
                            brand=brand,
                            screen=random.choice([6.1, 6.3, 6.5, 6.7]),
                            screen_type=random.choice(["OLED", "LTPO OLED", "AMOLED"]),
                            resolution=random.choice(["2556x1179", "2796x1290", "3200x1440"]),
                            refresh=random.choice([60, 90, 120]),
                            brightness=random.choice([1000, 1500, 2000]),
                            color=random.choice(["P3", "DCI-P3"]),
                            chip=random.choice(["A17 Pro", "骁龙8 Gen3", "天玑9300"]),
                            process=random.choice([3, 4, 5]),
                            cpu_cores=random.choice([6, 8]),
                            gpu=random.choice(["Apple GPU", "Adreno 750"]),
                            npu=random.choice(["Neural Engine", "NPU"]),
                            ram=random.choice([8, 12, 16]),
                            storage=random.choice([128, 256, 512]),
                            expand=random.choice(["不支持", "支持最大1TB"]),
                            main=random.choice([48, 50, 64, 108]),
                            ultra=random.choice([12, 48]),
                            tele=random.choice([5, 10, 12]),
                            front=random.choice([12, 16, 32]),
                            video=random.choice([4, 8]),
                            stabilization=random.choice(["OIS", "传感器位移式OIS"]),
                            battery=random.choice([3000, 4000, 4500, 5000]),
                            charge=random.choice([20, 30, 45, 66, 80]),
                            wireless=random.choice([15, 50]),
                            endurance=random.choice([15, 18, 20, 24]),
                            bluetooth=random.choice(["5.3", "5.4"]),
                            waterproof=random.choice(["IP68", "IP67", "IP65"]),
                            fingerprint=random.choice(["屏下指纹", "侧边指纹"]),
                            dimensions=random.choice(["146.6x70.6x7.8", "160.7x77.6x8.25"]),
                            weight=random.choice([170, 180, 200, 220]),
                            material=random.choice(["钛金属", "铝合金", "玻璃"]),
                            colors=random.choice(["黑、白、蓝", "深空黑、银、金"]),
                        )

                        keywords = [brand, product, category]
                        keywords.extend([w for w in title.split() if len(w) > 1])

                        doc = DocumentItem(
                            id=f"doc-{doc_id:04d}",
                            title=title,
                            content=content,
                            category=category,
                            product=product,
                            brand=brand,
                            keywords=list(set(keywords)),
                        )
                        documents.append(doc)
                        doc_id += 1

                        if len(documents) >= count:
                            return documents

    return documents


def save_data(faqs: list[FAQItem], documents: list[DocumentItem], output_dir: Path):
    """Save generated data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAQs
    faqs_file = output_dir / "faqs.json"
    with open(faqs_file, "w", encoding="utf-8") as f:
        json.dump([asdict(faq) for faq in faqs], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(faqs)} FAQs to {faqs_file}")

    # Save documents
    docs_file = output_dir / "documents.json"
    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump([asdict(doc) for doc in documents], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(documents)} documents to {docs_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate electronics product data")
    parser.add_argument("--faqs", type=int, default=500, help="Number of FAQs to generate")
    parser.add_argument("--docs", type=int, default=200, help="Number of documents to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated",
        help="Output directory",
    )

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output

    print(f"Generating {args.faqs} FAQs...")
    faqs = generate_faqs(args.faqs)

    print(f"Generating {args.docs} documents...")
    documents = generate_documents(args.docs)

    print(f"Saving data to {output_dir}...")
    save_data(faqs, documents, output_dir)

    print("Done!")

    # Print statistics
    print("\nStatistics:")
    print(f"  FAQs: {len(faqs)}")
    print(f"  Documents: {len(documents)}")

    # Category breakdown
    categories = {}
    for faq in faqs:
        categories[faq.category] = categories.get(faq.category, 0) + 1
    print("\nFAQ Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()