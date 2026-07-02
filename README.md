```
#HumanFaceStructuralValidator v1.2.0

**AI人脸生成流水线内置质检**|确定性规则引擎|光学物理验证|即插即用中间件

许可证：MIT Python：>=3.8状态：v1.2.0生产版本|55/55测试用例通过

---

##单线定位

基于物理学的解决方案----------------------------------------------------------------------------------------------如果没有通过验证就不能发货.

**此引擎定位为AI面部生成管道的内置质量检查员。“qualified=True”表示QC已通过并准备交付，而不是指面部属于真正的个人。**

---

##这解决了什么问题？

76%的用户认为数字人“令人毛骨悚然”--41%的用户是空视，33%的用户是机械动作[1]。这不是审美问题，而是物理结构问题，根据2025-2026年多份行业报告显示，2026年全球AI数字人市场规模已突破1000亿人民币[3]，但体验天花板仍被“恐怖谷”效应牢牢封住。

长期以来，人工智能生成的脸都有一种难以形容的“虚假”、“怪异”和“错误”的感觉。业界将此归因于“恐怖谷”，但解释仍停留在心理层面：他们可以告诉工程师“用户会感到不舒服”，但无法指出哪一种模式逻辑是错误的。这直接限制了数字人、虚拟流媒体、游戏NPC和电影后期制作的体验上限。

同时，行业当前的评估框架--由像素分布统计(如FID和LPIPS)主导--完全无视照明和结构可见度的物理自洽性。评估分数的提高不一定与物理真实性的提高相关。该引擎直接填补了缺失的“物理QC维度”。

HumanFaceStructuralValidator提供了一个工程上的答案：神秘谷效应从根本上是由AI脸违反了人类潜意识中嵌入的“活人面部物理验证器”造成的。该项目将潜意识规则集转化为可量化、可编程、确定性的公理系统，部署为生成管道中的预过滤器，以拦截物理上不兼容的面。2026年1月，米兰比科卡大学的一项研究独立地证实了当人类无法从行为上将人工智能面孔与真实面孔区分开来时，大脑对人工智能面孔保持着隐性的神经敏感性[2]--这是本研究中描述的“潜意识判断”的生理学基础。

---

##核心创新：先架构后部件

人类的视觉系统不会通过检查面部特征来判断“这是一张脸吗？”它首先检查的是整个骨骼结构--眼眶是否凹陷，鼻梁中线是否完整，颧骨和下颌轮廓是否属于人类。当框架完好无损时，即使部分缺失的特征(例如，一只眼睛，缺失的鼻子)也会被直观地感知为“人类”。当框架倒塌时，即使是完整的特征集也会被感知为“人类”。感觉不对。

基于这一发现，该项目做了四件事：

1.物理根本原因发现：将神秘谷效应追溯到违反四条铁律-肌肉张力、运动逻辑、结构完整性和光学物理学-并揭示现有像素统计度量(FID等)对物理自洽性的系统性盲目性。
2、判断公理体系：建立四层漏斗-骨骼框架→基本拓扑→比例+双中心→13项静态生理适应(包括光学)--每一层由硬数学规则强制执行。
3.光物理验证层：首次将光源与骨骼的交互作用纳入判断，基于朗伯余弦定律实时计算每个区域的亮度上限，拦截物理上不可能的“全显光”和“阴影区细节幻觉”。
4.开源中间件交付：提供对大型视觉模型零依赖的确定性规则引擎，可直接嵌入到生成管道中进行结构验证。

---

##前提条件：如何获取结构化测量数据

该引擎是一个纯粹基于规则的验证层，不包含图像处理模块。它需要一个上游特征提取器来将人脸图像转换为“StructuralData”结构。

推荐方法(开源、完全离线)：

-MediaPipe面部网格+自定义几何计算：输出478个面部标志，覆盖眼眶、鼻梁和轮廓。简单的几何计算(距离、曲率、对称性)可以组合所需的“结构数据”。

或者，可以使用商业面部地标API(例如，Face++、百度AI)来提取地标并组装数据结构。

建筑学：

```

原始图像→[您的功能提取器]→结构数据→[此引擎]→判断结果

```

欢迎社区提供基于MediaPipe、OpenCV、dlib等的适配器实现。有关适应指南，请参阅"docs/adapter_guidel.md"。

⚠ 风险说明：发动机的判断完全取决于上游“StructuralData”的准确性。如果上游提取器在极端俯仰角、严重遮挡或非真实感渲染风格下遇到界标漂移，则可能会发生错误判断。生产部署应使用“low_confidence”标记机制并预过滤极端输入。

---

##安装

"'bash
管道安装-e.
```

或从源安装：

```猛击
吉特克隆https://github.com/HeZijie/HumanFaceStructuralValidator.git
CD HumanFaceStructuralValidator
管道安装-e.
```

依赖项：Python3.8+，零第三方包.

---

快速入门

5分钟演示

```猛击
#运行全部55个测试用例
Python HumanFaceStructuralValidator.py
```

预期输出：55个测试用例的判断结果，所有匹配预期结果。

手动数据构造示例

```python
从human_face_validator导入HumanFaceStructuralValidator、StructuralData

#手动构建最小结构化数据(仅演示)
data=StructuralData(
orbit_socket_depth=0.8，
orbit_position_upper_third=真，
鼻_中线_连续=真，
jaw_symmetry=真，
has_single_closed_contour=真，
global_symmetry_pass=True，
eye_upper_third=真，
nose_midline_center=真，
口中心下方=真，
形态_适应={
“eye_system”：0.93，“nose_system”：0.93，“lip_system”：0.92，
“jaw_system”：0.92，“contour_system”：0.93
    },
visual_focus_system="eye_system"，
美学_焦点_系统="eye_system"，
has_multiple_visual_focus=False，
has_multiple_aetestic_focus=False，
眼张力mm=1.5，
皮肤配合骨=真，
张力_统一=真，
skin_texture_ratio=0.95，
蒙皮_锐度=0.70，
global_face_brightness=0.65，
    shadow_area_ratio=0.18,
    skin_region_a_stdev=0.05,
    texture_direction_variance=0.30,
    # Optical validation fields (optional; auto-skipped if not provided)
    light_azimuth=45.0,
    light_pitch=45.0,
    light_intensity=1.0,
    region_normal_light_cosine={},
    region_measured_brightness={},
    region_edge_gradient={},
)

validator = HumanFaceStructuralValidator()
result = validator.validate(data)

print(result.qualified)      # True
print(result.reject_reason)  # None
```

📌 In production, StructuralData should be automatically populated by an upstream feature extractor (e.g., MediaPipe). The above is for demonstrating the data structure only.

---

Project Structure

```
HumanFaceStructuralValidator/
├── HumanFaceStructuralValidator.py                                    # Main Engine v1.2.0
├── AI-Generated Faces and the Uncanny Valley.md                       # English Whitepaper
├── AI生成人脸恐怖谷效应白皮书-v1.2.1.md                               # Chinese Whitepaper
├── docs/
│   └── adapter_guide.md                                               # Upstream Adapter Guide
├── LICENSE                                                            # MIT License
└── README.md
```

---

Judgment Pipeline

```
Input StructuralData
        ↓
  [Layer 0] Skeletal Framework Intact?
    ├─ No → topology_abnormal (hard reject)
    └─ Yes ↓
  [Layer 1] Basic Topology Present?
    ├─ No → basic_topology_fail (hard reject)
    └─ Yes ↓
  [Layer 2] Proportion + Dual-Center ≥ 70%?
    ├─ No → proportion_model_fail (hard reject)
    └─ Yes ↓
  [Layer 3] All 13 Physical Iron Laws Passed?
    ├─ No → tension_fail / system_invasion / ... (specific reason returned)
    └─ Yes → ✅ QC Passed, Ready for Delivery
```

第0层：骨骼框架完整性
确认眶窝凹陷、连续鼻桥中线和下颌对称。任何失败→拓扑结构异常，硬排斥。

第1层：基本拓扑存在性
验证单个闭合轮廓、整体对称、倒三角拓扑(眼睛在上1/3，鼻子沿中线，嘴在下居中)。failure→basic_topology_fail，硬拒绝。

第2层：感知比例模型匹配+双中心唯一性
根据视觉中心度，将面部分为七个模块(眼睛、鼻子、嘴唇、下巴、前额、下巴、脸颊)，并进行分级公差。同时检查视觉中心和美学中心是否各不相同-分散的双中心会产生零校正加值，需要原始匹配分数才能在没有辅助的情况下清除70%阈值。失败→比例_模型_失败。

第3层：13项静态生理适应(在v1.2.0中扩展为13项)
通过前两层的面进行13次物理铁法检查：

1.整体形态适应度≥65%，单模块偏差≤8%
2.眼张力：上眼睑覆盖上角膜缘1-2mm(额角/轻微轮廓；极限角度自动减重)
3.皮肤-肌肉张力：软组织符合纵向骨骼轮廓
4.整体张力均匀度：所有区域张力节律一致
5.纵向轮廓平滑度：相邻区域轮廓差≤±2步，无孤立的尖峰或滴
6.皮肤纹理强度≥真实人体样本库最小值的70%
7.系统间入侵：任意面部特征模块入侵相邻系统标准边界>30%→失败
8.皮肤锐度：基于人体对比敏感度函数生理光学上限(≤0.82)
9.光影分层：整体亮度≤0.88，阴影面积比≥12%
10.血红蛋白色调波动：皮肤区域a*标准差≥0.03
11.纹理方向方差：区域间蒙皮沟槽方向方差≥0.15
12.解剖边缘梯度动态检查：边缘梯度上限随光源角度实时调整；强烈掠射光线下严禁锐化纹理
13.骨连接阴影深度检查：通过Lambert余弦定律计算的每区域理论亮度上限；测量的亮度超过1.2×理论最大值触发拦截

任何一次失败都会返回一个特定的reject_reason来指导目标再生。13次都通过，即为QC合格状态。

---

面部光学一致性验证框架(v1.2.0中统一)

规则6、8、9、10和11构成五维皮肤判断系统，规则12和13用光学物理约束来补充该系统。所有七个规则共享一个单一的基本逻辑：生成的面部必须同时满足五个维度上的真实人类光学特性-纹理强度、锐度上限、光/影分层、血红蛋白色调波动和纹理方向变化-并且在任何给定光源下不得违反朗伯定律和骨遮挡逻辑。

核心配方(新)：

·区域理论亮度上限：L_max=I·(N·L)·(1-k·d)
·结构可见度：C_感知=C_物理·exp(−(θ−90°)²/(2σ²))·M(d)

如果AI强制渲染眼眶阴影区的纹理，或在凹陷区域(如鼻根或眼窝)产生不可能的高光，则光学检查直接截取，统一标记为over_smooth_skin(可细分为光学子类型)。所有阈值均来源于解剖原理和认知科学研究，有待大规模实像经验校准，理论依据见白皮书附录B。

---

使用案例

· 🎮 游戏/虚拟人：在批量生成NPC时自动拦截“假笑”和“瞪眼”人脸
· 🎬 胶片后期制作：AI换面或帧插值后的结构柔度QC
· 🛡️ 内容调节：识别deepfake工作面中的结构异常(结合其他解决方案)
· 🎨  AI艺术工具：自动过滤扩散模型输出，仅提供“实体活”的面部

---

性能

⚡ 理论估算(纯数学比较，无I/O，无模型推断)：

·每面判断<0.1ms(仅规则计算，不包括上游特征提取时间)
·新的光学物理公式仅涉及浮点乘法/除法和三角函数，没有额外的计算开销；理论上的每次判断时间<0.1ms
·理论上可在Raspberry Pi和移动终端上部署
·实际吞吐量取决于上游特征提取器速度

注：以上为理论估计数。欢迎社区提供的真实基准。

---

代表性测试用例

以下是55个案例测试套件中的代表性案例。完整的测试报告请参见白皮书v1.2.1和HumanFaceStructuralValidator.Py run_final_tests()。

# Input Result Triggered Rule
1 Normal adult, frontal ✅ Qualified Dual-center unique, +10% correction
2 AI staring eyes ❌ Rejected tension_fail (abnormal eye tension)
3 Silicone mask (insufficient texture) ❌ Rejected over_smooth_skin
4 AI composite face (multi-module collapse) ❌ Rejected proportion_model_fail + scattered dual-center
5 Real single-eye missing (framework intact) ✅ Qualified Framework intact, dual-center correction clears threshold
6 AI cyclops monster (no orbital socket + misplaced eye) ❌ Rejected topology_abnormal (skeletal framework collapse)
7 Infant, frontal ✅ Qualified Center correction clears threshold
8 AI face with globally uniform, bloodless skin tone ❌ Rejected over_smooth_skin (insufficient hemoglobin hue fluctuation)
9 AI face with uniformly stretched texture direction ❌ Rejected over_smooth_skin (texture direction variance too low)
10 Correct proportions but inter-system invasion ❌ Rejected system_invasion (nose invades eye boundary >30%)
11 Nasal base anomalous brightening (violates Lambert's law) ❌ Rejected over_smooth_skin (light/shadow logic anomaly)
12 Shadow zone texture excessively sharp (structural visibility anomaly) ❌ Rejected over_smooth_skin (shadow zone gradient exceeds limit)

Case 5 validates "Framework Before Parts": a real person with a missing eye passes Layer 0 because the skeletal framework is intact; the AI monster face is intercepted at Layer 0 due to missing orbital depression and twisted midline. The difference lies in the framework, not organ count. Cases 11–12 demonstrate how optical validation intercepts physically impossible generation results.

---

技术特点

·确定性规则引擎：不接触原始图像，不调用任何大型视觉模型，只接受结构测量数据，纯数学逻辑计算。
·完全离线，数据不会离开现场：无互联网，无云API调用。适用于金融、政府和其他数据合规性场景。
·白盒解释性：每次拒绝都提供一个物理原因(例如，张力失效、系统侵入、皮肤过度光滑)，可直接用于生成模型反馈。
·超低计算要求：纯数学，无需GPU。理论单判断<0.1ms，可部署在标准服务器甚至移动终端上。
·对大型模型的补充：大型模型的回答是“它看起来像真的吗？”；此引擎回答“物理上正确吗？”
·学术上可追溯：所有阈值均可追溯到独立的认知科学研究(Tsao等人，2006-2010年，Tanaka&Farah1993年，Maurer等人2002年，Itti&Koch2001年，Campbell&Robson1968年，Anderson&Parrish1981年，Pierard等人2000年，Lambert1760年，Debevec&Malik1997年，Ruderman&Bialek1994年)。

---

与现有解决方案的关系

此引擎定位为生成管道中的预过滤器结构验证层。它是对其他解决方案的补充，而不是替代：

解原理解释关系
大型视觉模型鉴别器深度学习黑盒低(仅概率)补充：发动机预过滤器，大型模型后审核
活动检测(3D结构光)硬件深度/IR信息中等补充：活动检查“真实人脸”；引擎检查“生成的脸看起来是否活动”
Facial Landmark APIs Geometric feature extraction Medium Upstream-downstream: API provides structural data, engine consumes it for validation
This Engine Physical structure rule validation White-box, precise to mm/percentage —

---

Integration

```python
from human_face_validator import HumanFaceStructuralValidator

# 1. Upstream feature extractor converts image to structural data
structural_data = upstream_extractor.extract(input_image)

# 2. Rule engine performs deterministic validation
validator = HumanFaceStructuralValidator()
result = validator.validate(structural_data)

# 3. Branch based on judgment result
if result.qualified:
    aesthetic_engine.process(input_image)          # Passed, proceed to downstream pipeline
else:
    handle_non_human(result.reject_reason)         # Rejected, guide model regeneration
```

---

Output Interface

Pass example:

```json
{
  "qualified": true,
  "confidence": "high",
  "reject_reason": null,
  "risk_flags": {
    "horror_valley_risk": false,
    "structural_conflict": false,
    "dual_center_scattered": false,
    "partial_feature_risk": false,
    "low_confidence": false,
    "non_typical_human_proportion": false,
    "system_invasion_detected": false
  }
}
```

拒绝示例：

```JSON
{
“合格”：false，
"置信度"："高"，
"拒绝原因"："皮肤过度光滑"，
"risk_flags"：{
"恐怖谷风险"：是真的，
"structural_conflict"：false，
“dual_center_scattered”：false，
"partial_feature_risk"：false，
"low_confidence"：false，
“非典型人比例”：false，
"system_invention_detected"：false
  }
}
```

置信规则：

·合格时：真，置信度可为高、中或低；
·合格时：假，置信度默认为高(结构缺陷具有确定性)，除非触发low_confidence标志；
·如果low_confidence为真，则无论判断结果如何，置信度都将降至低，建议手动审查合格：真实结果。

---

版本说明

当前版本：v1.2.0(生产版本)，对应于白皮书v1.2.1(文档修订).

v1.2.0主要变更：
·第3层从11个检查扩展到13个检查：增加了解剖边缘梯度动态检查和骨链阴影深度检查。
·建立了面部光学一致性验证框架，引入了朗伯余弦定律和结构可见度动态阈值公式。
·添加行业评估指标批评，确定FID/LPIPS对物理自一致性的系统性盲目性。
·测试用例从47个扩展到55个，涵盖光学违规场景。

历史版本(v1.0.1、v1.1.0)的代码和白皮书归档在项目的提交历史中。此存储库仅保留最新的生产版本。

---

路线图

·v1.2.0(已发布) ✅：光学物理验证(朗伯阴影深度、结构可见性动态阈值)
·v1.3.0(已划分)：镜面反射/次表面散射扩展、多光源叠加、官方中管适配器、轮廓视图判断规则
·v2.0(长期)：动态表情时间验证、肌肉运动逻辑验证、完整的端到端流水线(图像输入→结构提取→判断输出)

---

FAQ

问：为什么不用一个大模型来判断呢？
a：大型模特的回答是“它看起来像真的吗？”-主观概率。该引擎回答“物理上正确吗？”-确定性规则。两者是互补的，并且该引擎具有零推理成本，使其成为批量QC的理想引擎。

问：2D关键点输入引入了多少错误？
答：2D输入不能直接计算3D曲率，触发low_confidence标志。判断结果仅供参考。建议使用3D关键点(例如，MediaPipe的478个带深度信息的点)以获得最佳精度。

问：这能用来鉴别真假脸/检测deepfakes吗？
答：此引擎定位为静态结构验证层-防御系统中的第一个组件，而不是完整的真实性鉴别器。生产部署应将其与动态生物特征验证(rPPG、微表达)、活动检测和内容来源技术相结合。

问：它对动漫/卡通人物脸有效吗？
答：目前的规则集是为真实人体解剖结构设计的，不适用于非真像风格。未来可能会探索一个风格化的适应模块。

问：这些规则的依据是什么种族/年龄组？
答：目前的阈值是基于一般成人的解剖结构。跨种族和跨年龄组的专门适应是未来迭代路线图的一部分。

问：这与Face++或百度AI的面部质量评估有何不同？
答：一般面部质量评估评估的是清晰度、光照、姿势等通用指标。这个引擎专门针对“恐怖谷效应”，判断面部的物理结构是否符合活人的铁律。两者在不同的维度上操作，可以互补使用。

问：为什么输出字段从real_face改为合格的？
答：旧的real_face字段很容易被误解为验证图像是真实照片还是AI生成的引擎。在现实中，我们对生成管道进行结构质量检查--判断AI生成的脸是否有生命。合格更准确地传达“QC通过，准备交付”。

问：光学验证还需要哪些其他数据？
答：上游必须提供光源方向、每个区域的法向量、每个区域的测量亮度，如果不提供，则在不影响其他规则的情况下自动跳过光学验证，在生产中可以通过MediaPipe+简单的光源估计获得这些字段。

问：对于复杂的蒙皮材质(镜面反射高光、半透明)，朗伯模型是否足够？
a:v1.2.0使用Lambertian扩散近似+静态骨封堵，这已经覆盖了大多数生成缺陷。在v1.3.0中计划了镜面反射和次表面散射扩展。

---

许可证分层说明

·源代码(HumanFaceStructuralValidator.py等)：MIT许可证。免费用于商业用途、修改和重新分发。白皮书附录D代码与此存储库代码共享相同的MIT许可证。
·白皮书文本和判断公理规则系统(阈值、层结构和判断逻辑的组合)：保留所有权利。在产品中商业使用完整的规则系统需要获得作者的单独书面许可。
·学术引用、个人学习、非商业性研究：完全允许仅有归属。

💡 纯语言：代码可以自由使用、修改和商业化。但是，如果您想以完整的规则体系--“四层判断+双中心+30%系统边界阈值+光学物理公式”--并将其包装为商业产品或服务，请联系作者进行许可。

概念验证声明：已验证模拟结构数据的逻辑闭环(55/55个测试用例通过)，涵盖所有真人类型、常见AI缺陷和新增的光学物理违规场景(鼻底异常增亮、阴影区梯度超标等)。从真实像素到结构数据的端到端精度尚未在大规模真实图像上校准。生产部署必须与动态生物特征验证、活性检测、以及内容来源技术。尽管有此通知，但单独部署此引擎意味着用户承担所有由此产生的业务风险。

---

参考文献

[1]斯坦福大学(2022年)。数字人和虚拟化身的用户体验研究：报告的怪异感觉及其按特征类别的分解。

[2]米兰比科卡大学(2026年)。超真实AI生成的面孔的神经特征：从隐神经评价中分离行为的不可区分性。

[3]多家咨询公司2025-2026全球人工智能数字人/虚拟人市场研究报告(综合引用)。

[4]Lambert，J.H.(1760).《光度计》.

[5]Debevec，P.E.和Malik，J.(1997年)。从照片中恢复高动态范围辐射亮度图。

[6]鲁德曼，D.L.，&Bialek，W.(1994).自然图像的统计：森林中的缩放.物理评论快报.

---

作者

贺子杰，“统一面部视觉美学系统”的创始人，长期从事面部生成美学评价和结构顺应性研究，发表了《人工智能生成的面部与恐怖谷》白皮书。

---

贡献的

欢迎问题和拉请求。

最需要的贡献方向：

·MediaPipe面部网格适配器：将MediaPipe的478个面部界标映射到HumanFaceStructuralValidator.Py的结构数据中定义的曲率、距离、对称性和光学几何参数。将PR提交到适配器/目录。有关详细信息，请参见docs/adapter_guidel.md。
·dlib/OpenCV备用适配器：将其他地标检测解决方案的输出映射到结构数据。
·测试用例扩展：为纵断面图、遮挡、非真实感风格和其他照明条件光学验证用例添加边缘用例测试数据。
·极端角度鲁棒性验证：验证上游界标漂移下低置信度标志触发的准确性。

如需商务咨询，请通过项目问题或直接消息联系。

---

恐怖谷·生理结构适应·肌肉运动逻辑·QC中间件·开源中间件·面部质量评估·Deepfake检测·虚拟数字人类·扩散模型·光学物理验证

人工智能的脸到底在哪里看起来是“假的”？我们已经精确定位--精确到毫米、百分比以及违反兰伯特定律的精确坐标。

```
