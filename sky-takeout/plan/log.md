# Day 00: 环境配置与尝试

## 1. IDE 与数据库准备
- 先下ide吧
- intelj mysql reddit？
- 我先官网下了一个intelj
- 下载mysql admin密码 创建账号者 密码 权限
- 已经添加好环境变量
- `mysql.exe -u root -p密码`

### 数据库初始化 SQL
```sql
CREATE DATABASE IF NOT EXISTS sky_takeout CHARACTER SET utf8mb4;
USE sky_takeout;

DROP TABLE IF EXISTS `employee`;
CREATE TABLE `employee` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `name` varchar(32) COLLATE utf8_bin NOT NULL COMMENT '姓名',
  `username` varchar(32) COLLATE utf8_bin NOT NULL COMMENT '用户名',
  `password` varchar(64) COLLATE utf8_bin NOT NULL COMMENT '密码',
  `phone` varchar(11) COLLATE utf8_bin NOT NULL COMMENT '手机号',
  `sex` varchar(2) COLLATE utf8_bin NOT NULL COMMENT '性别',
  `id_number` varchar(18) COLLATE utf8_bin NOT NULL COMMENT '身份证号',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态 0:禁用，1:启用',
  `create_time` datetime NOT NULL COMMENT '创建时间',
  `update_time` datetime NOT NULL COMMENT '更新时间',
  `create_user` bigint NOT NULL COMMENT '创建人',
  `update_user` bigint NOT NULL COMMENT '修改人',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb3 COLLATE=utf8_bin COMMENT='员工信息';

INSERT INTO `employee` VALUES (1, '管理员', 'admin', 'e10adc3949ba59abbe56e057f20f883e', '13800000000', '1', '110101199001010047', 1, NOW(), NOW(), 1, 1);
```

### 常用命令尝试
```sql
SHOW DATABASES;

USE sky_takeout;

SHOW TABLES;

SELECT * FROM employee;

SELECT username, password FROM employee;

SELECT * FROM employee WHERE username = 'admin';

DESC employee;

DROP TABLE employee;

DROP DATABASE sky_takeout;
```

## 2. Maven 与 Project 创建
- intelj maven jdk21 创建一个project
- `explorer $env:USERPROFILE` 配置镜像源
- 创建 `.m2`
- `settings.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd">
    <mirrors>
        <mirror>
            <id>aliyunmaven</id>
            <mirrorOf>*</mirrorOf>
            <name>阿里云公共仓库</name>
            <url>https://maven.aliyun.com/repository/public</url>
        </mirror>
    </mirrors>
</settings>
```

## 3. 遇到问题与解决过程
- 用maven一顿操作不行 但是最开始创建选java就能一下子箭头运行成功有打印？
- 不对啊 用的是maven 又新建一个就好了？

## 4. 依赖与验证
- 在 `</properties>` 和 `</project>` 标签之间，插入下面这段 `<dependencies>` 代码：

```xml
    <dependencies>
        <dependency>
            <groupId>com.alibaba.fastjson2</groupId>
            <artifactId>fastjson2</artifactId>
            <version>2.0.47</version>
        </dependency>
    </dependencies>
```

- 新main代码：

```java
package org.example;

// 导入我们刚刚下载的第三方包里的类
import com.alibaba.fastjson2.JSONObject;

public class Main {
    public static void main(String[] args) {
        // 创建一个 JSON 对象（类似 Python 里的字典）
        JSONObject myJson = new JSONObject();
        myJson.put("projectName", "sky-takeout");
        myJson.put("developer", "Aier");
        myJson.put("status", "Maven is working perfectly!");

        // 打印出来看看效果
        System.out.println("生成的 JSON 数据：");
        System.out.println(myJson.toJSONString());
    }
}
```

- 新main代码 更新环境包 等一会 运行不报错了 正常运行
- maven和 外部库可以看到

---

# 苍穹外卖核心突击：4天极速通关计划

> **说明：** 第0天已完成环境配置、数据库初始化与基础工具的测试。后续针对“苍穹外卖”项目，整理了这份高强度的4天突击计划，过滤掉冗杂的边缘功能，直击后端开发的核心技术痛点与必考知识点。

## Day 1：骨架成型与用户守门（架构搭建与登录逻辑）
**核心目标：** 把多模块工程跑起来，打通第一个也是最重要的**员工登录**链路。
- **后端架构初始化：**
  - 导入 `sky-common`、`sky-pojo`、`sky-server` 多模块基础骨架。
  - 配置 `application.yml` 中 MyBatis、数据库以及常量等。
- **员工登录链路开发：**
  - 理清 DTO（入参接收）、Entity（数据库映射）、VO（出参返回）的区别及应用。
  - 密码的 MD5 加密比对。
- **登录鉴权大闸：**
  - JJWT 生成 Token ，并编写 Spring MVC **拦截器**。
  - 使用 `ThreadLocal` 在当前线程内保存解析出的员工 ID。
- **规范与工具：**
  - 配置**全局异常处理器**统一返回错误类型。
  - 配置 Swagger / Knife4j 接口文档，使用接口文档进行登录测试。

## Day 2：管理后台的“增删改查”进阶与黑科技（AOP与OSS）
**核心目标：** 拿下后台核心表数据的处理，学习企业中常用的自动化配置和第三方服务接入。
- **员工管理模块开发：**
  - 基于 `PageHelper` 插件完成员工分页查询。
  - 员工详情查询、信息编辑与账号启用/禁用。
- **AOP 公共字段自动填充（重点面试题）：**
  - 背景：避免每次写入数据库都要手动 `setCreateTime` 与 `setUpdateUser`。
  - 实现：自定义注解 `@AutoFill`，配合 AOP 前置通知与 Java 反射自动为实体类填充属性。
- **文件上传服务：**
  - 菜品图片上传：打通阿里云 OSS（或基于本地磁盘的文件上传配置）。
- **菜品分类及简单业务：**
  - 分类数据的增删改查。

## Day 3：复杂业务组合拳与性能第一关（多表事务与Redis缓存）
**核心目标：** 应对真实的业务复杂度，解决多表操作的数据一致性难题，并引入缓存加速查询。
- **菜品管理的高难度操作：**
  - 新增菜品（涉及跨表事务：`dish` 菜品表 + `dish_flavor` 口味表），必须掌握 `@Transactional` 事务管理。
  - 菜品的起售/停售状态切换，以及信息的修改与删除。
- **套餐管理（Setmeal）：**
  - 梳理套餐与菜品的“多对多”关联关系。
  - 套餐的新增与展示逻辑。
- **Redis 缓存实战与性能优化：**
  - 启动本地 Windows/Docker Redis 环境配置。
  - 使用 Spring Data Redis 实现菜品缓存列表（提高用户端查询速度）。
  - 处理“缓存一致性”问题：后台修改菜品数据时，切记清理对应 Redis 缓存。

## Day 4：微信小程序对接，下单流转与消息实录
**核心目标：** 接入C端(消费者端口)，完成一套标准的电商/外卖购物车与下单流程。
- **微信小程序端接入：**
  - 掌握微信登录的免密鉴权流程，基于小程序端调用 `wx.login` 获取 code ，微信接口换取 OpenID 入库。
- **购物车逻辑（DB/Redis方案）：**
  - 加入购物车、清空购物车、展示购物车商品（可以练习使用 Redis 的 Hash 结构进行替换）。
- **用户下单与支付模拟：**
  - 订单核心逻辑：校验参数、生成订单号、扣库存、记录订单明细（`order`与`order_detail`表）。
  - 模拟微信支付成功后的订单状态流转与回调处理。
- **进阶特性与收尾：**
  - **Spring Task 定时任务**：实现“支付超时自动取消订单。
  - **WebSocket 实时通信（可选）**：从零搭建WebSocket服务，实现“您有新的外卖订单，请注意查收”的商家端语音播报功能。