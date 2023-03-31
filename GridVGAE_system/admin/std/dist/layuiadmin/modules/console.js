/** layuiAdmin.std-v1.2.1 LPPL License By http://www.layui.com/admin/ */
;layui.define(function (e) {
    layui.use(["admin", "carousel"], function () {
        var e = layui.$, t = (layui.admin, layui.carousel), a = layui.element, i = layui.device();
        e(".layadmin-carousel").each(function () {
            var a = e(this);
            t.render({
                elem: this,
                width: "100%",
                arrow: "none",
                interval: a.data("interval"),
                autoplay: a.data("autoplay") === !0,
                trigger: i.ios || i.android ? "click" : "hover",
                anim: a.data("anim")
            })
        }), a.render("progress")
    }), layui.use(["admin", "carousel", "echarts"], function () {
        var e = layui.$, t = layui.admin, a = layui.carousel, i = layui.echarts, l = [], n = [{
            title: {text: "生成构型分布", x: "center", textStyle: {fontSize: 14}},
            tooltip: {trigger: "item", formatter: "{a} <br/>{b} : {c} ({d}%)"},
            legend: {orient: "vertical", x: "left", data: ["Ising", "XY", "Potts", "Other"]},
            series: [{
                name: "访问来源",
                type: "pie",
                radius: "55%",
                center: ["50%", "50%"],
                data: [{value: 9052, name: "Ising"}, {value: 1610, name: "XY"}, {
                    value: 3200,
                    name: "Potts"
                }, {value: 1700, name: "Other"}]
            }]
        }, {
            title: {text: "最近一周生成构型数量", x: "center", textStyle: {fontSize: 14}},
            tooltip: {trigger: "axis", formatter: "{b}<br>新增构型：{c}"},
            xAxis: [{type: "category", data: ["3-07", "3-08", "3-09", "3-10", "3-11", "3-12", "3-13"]}],
            yAxis: [{type: "value"}],
            series: [{type: "line", data: [20000, 30000, 40000, 61000, 15000, 27000, 38000]}]
        }], r = e("#LAY-index-dataview").children("div"), o = function (e) {
            l[e] = i.init(r[e], layui.echartsTheme), l[e].setOption(n[e]), t.resize(function () {
                l[e].resize()
            })
        };
        if (r[0]) {
            o(0);
            var d = 0;
            a.on("change(LAY-index-dataview)", function (e) {
                o(d = e.index)
            }), layui.admin.on("side", function () {
                setTimeout(function () {
                    o(d)
                }, 300)
            }), layui.admin.on("hash(tab)", function () {
                layui.router().path.join("") || o(d)
            })
        }
    }), layui.use("table", function () {
        var e = (layui.$, layui.table);
        e.render({
            elem: "#LAY-index-topSearch",
            url: layui.setter.base + "json/console/top-search.js",
            page: !0,
            cols: [[{type: "numbers", fixed: "left"}, {
                field: "keywords",
                title: "关键词",
                minWidth: 300,
                templet: '<div><a href="https://www.baidu.com/s?wd={{ d.keywords }}" target="_blank" class="layui-table-link">{{ d.keywords }}</div>'
            }, {field: "frequency", title: "搜索次数", minWidth: 120, sort: !0}, {
                field: "userNums",
                title: "用户数",
                sort: !0
            }]],
            skin: "line"
        }), e.render({
            elem: "#LAY-index-topCard",
            url: layui.setter.base + "json/console/top-card.js",
            page: !0,
            cellMinWidth: 120,
            cols: [[{type: "numbers", fixed: "left"}, {
                field: "title",
                title: "标题",
                minWidth: 300,
                templet: '<div><a href="{{ d.href }}" target="_blank" class="layui-table-link">{{ d.title }}</div>'
            }, {field: "username", title: "发帖者"}, {field: "channel", title: "类别"}, {
                field: "crt",
                title: "点击率",
                sort: !0
            }]],
            skin: "line"
        })
    }), e("console", {})
});