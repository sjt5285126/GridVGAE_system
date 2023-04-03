/** layuiAdmin.std-v1.2.1 LPPL License By http://www.layui.com/admin/ */
layui.define(["table", "form"], function (e) {
  var t = layui.$,
    i = layui.table;
  layui.form;
  i.render({
    elem: "#LAY-user-Dataset-manage",
    url: layui.setter.base + "json/useradmin/mangaDataset.js",
    cols: [
      [
        { type: "checkbox", fixed: "left" },
        { field: "id", width: 80, title: "ID", sort: !0 },
        {
          field: "Datasetname",
          title: "数据集名称",
        },
        { field: "datasetclass", title: "数据集类别" },
        { field: "jointime", title: "加入时间", sort: !0 },
        {
          field: "check",
          title: "是否检阅" /**可以不要这部分 */,
          templet: "#buttonTpl",
          minWidth: 80,
          align: "center",
        },
        {
          title: "操作",
          width: 150,
          align: "center",
          fixed: "right",
          toolbar: "#table-useradmin-admin",
        },
      ],
    ],
    text: "对不起，加载出现异常！",
  }),
    i.on("tool(LAY-user-Dataset-manage)", function (e) {
      e.data;
      /** 该处与后端密码结合 */
      if ("del" === e.event)
        layer.prompt(
          { formType: 1, title: "敏感操作，请验证密码" },
          function (t, i) {
            layer.close(i),
              layer.confirm("确定删除此数据集？", function (t) {
                console.log(e), e.del(), layer.close(t);
              });
          }
        );
      else if ("edit" === e.event) {
        /** 这部分可以不要可以大该 应该是需要的只是要大改 */
        t(e.tr);
        layer.open({
          type: 2,
          title: "编辑管理员",
          content: "../../../views/user/UpAndDown/UploadDataset.html",
          area: ["420px", "420px"],
          btn: ["确定", "取消"],
          yes: function (e, t) {
            var l = window["layui-layer-iframe" + e],
              r = "LAY-user-back-submit",
              n = t
                .find("iframe")
                .contents()
                .find("#" + r);
            l.layui.form.on("submit(" + r + ")", function (t) {
              t.field;
              i.reload("LAY-user-front-submit"), layer.close(e);
            }),
              n.trigger("click");
          },
          success: function (e, t) {},
        });
      }
    }),
    i.render({
      elem: "#LAY-user-back-role",
      url: layui.setter.base + "json/useradmin/role.js",
      cols: [
        [
          { type: "checkbox", fixed: "left" },
          { field: "id", width: 80, title: "ID", sort: !0 },
          {
            field: "rolename",
            title: "角色名",
          },
          { field: "limits", title: "拥有权限" },
          { field: "descr", title: "具体描述" },
          {
            title: "操作",
            width: 150,
            align: "center",
            fixed: "right",
            toolbar: "#table-useradmin-admin",
          },
        ],
      ],
      text: "对不起，加载出现异常！",
    }),
    i.on("tool(LAY-user-back-role)", function (e) {
      e.data;
      if ("del" === e.event)
        layer.confirm("确定删除此角色？", function (t) {
          e.del(), layer.close(t);
        });
      else if ("edit" === e.event) {
        t(e.tr);
        layer.open({
          type: 2,
          title: "编辑角色",
          content: "../../../views/user/administrators/roleform.html",
          area: ["500px", "480px"],
          btn: ["确定", "取消"],
          yes: function (e, t) {
            var l = window["layui-layer-iframe" + e],
              r = t.find("iframe").contents().find("#LAY-user-role-submit");
            l.layui.form.on("submit(LAY-user-role-submit)", function (t) {
              t.field;
              i.reload("LAY-user-back-role"), layer.close(e);
            }),
              r.trigger("click");
          },
          success: function (e, t) {},
        });
      }
    }),
    e("useradmin", {});
});
